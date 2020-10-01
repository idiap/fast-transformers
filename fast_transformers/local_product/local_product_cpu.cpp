//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
//

#include <limits>

#include <torch/extension.h>


inline float dot(const float *A, const float *B, int D) {
    float s = 0;
    for (int i=0; i<D; i++) {
        s += (*A) * (*B);
        A++;
        B++;
    }
    return s;
}


inline void scaled_copy_add(const float *src, float *dst, float scale, int D) {
    for (int i=0; i<D; i++) {
        *dst += (*src) * scale;
        dst++;
        src++;
    }
}


torch::Tensor local_dot_product(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor attn_mask,
    torch::Tensor key_lengths,
    int local_context
) {
    // Extract some shapes
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);

    // Allocate space for the output
    auto output = queries.new_full({N, H, L, local_context}, -1e24);

    // Create accessors for all the arguments
    auto qa = queries.accessor<float, 4>();
    auto ka = keys.accessor<float, 4>();
    auto oa = output.accessor<float, 4>();
    auto kla = key_lengths.accessor<int64_t, 1>();
    auto ama = attn_mask.accessor<float, 2>();

    #pragma omp parallel for collapse(2)
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            int n_keys = kla[n];
            for (int l=0; l<L; l++) {
                int start = std::max(0, l-local_context/2);
                int end = std::min(n_keys, l+(local_context+1)/2);
                int kstart = local_context/2 - std::abs(l-start);
                for (int s=start,k=kstart; s<end; k++, s++) {
                    oa[n][h][l][k] = dot(
                        &qa[n][h][l][0],
                        &ka[n][h][s][0],
                        E
                    ) + ama[l][s];
                }
            }
        }
    }

    return output;
}


std::tuple<torch::Tensor, torch::Tensor> local_dot_backward(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor key_lengths,
    torch::Tensor grad,
    int local_context
) {
    // Extract some shapes
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);

    // Allocate space for the output
    auto grad_queries = torch::zeros_like(queries);
    auto grad_keys = torch::zeros_like(keys);

    // Create accessors for all the arguments
    auto qa = queries.accessor<float, 4>();
    auto ka = keys.accessor<float, 4>();
    auto kla = key_lengths.accessor<int64_t, 1>();
    auto ga = grad.accessor<float, 4>();
    auto gqa = grad_queries.accessor<float, 4>();
    auto gka = grad_keys.accessor<float, 4>();

    // Compute the gradient for the queries
    #pragma omp parallel for collapse(2)
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            int n_keys = kla[n];
            for (int l=0; l<L; l++) {
                int start = std::max(0, l-local_context/2);
                int end = std::min(n_keys, l+(local_context+1)/2);
                int kstart = local_context/2 - std::abs(l-start);
                for (int s=start,k=kstart; s<end; k++, s++) {
                    scaled_copy_add(
                        &ka[n][h][s][0],
                        &gqa[n][h][l][0],
                        ga[n][h][l][k],
                        E
                    );
                    scaled_copy_add(
                        &qa[n][h][l][0],
                        &gka[n][h][s][0],
                        ga[n][h][l][k],
                        E
                    );
                }
            }
        }
    }

    return std::make_tuple(grad_queries, grad_keys);
}


torch::Tensor local_weighted_average(
    torch::Tensor attention,
    torch::Tensor values
) {
    // Extract some shapes
    int N = attention.size(0);
    int H = attention.size(1);
    int L = attention.size(2);
    int local_context = attention.size(3);
    int S = values.size(2);
    int E = values.size(3);

    // Allocate space for the output
    auto output = torch::zeros({N, H, L, E}, values.options());

    // Create accessors for all the arguments
    auto aa = attention.accessor<float, 4>();
    auto va = values.accessor<float, 4>();
    auto oa = output.accessor<float, 4>();

    #pragma omp parallel for collapse(2)
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            for (int l=0; l<L; l++) {
                int start = std::max(0, l-local_context/2);
                int end = std::min(S, l+(local_context+1)/2);
                int kstart = local_context/2 - std::abs(l-start);
                for (int s=start,k=kstart; s<end; k++, s++) {
                    scaled_copy_add(
                        &va[n][h][s][0],
                        &oa[n][h][l][0],
                        aa[n][h][l][k],
                        E
                    );
                }
            }
        }
    }

    return output;
}


std::tuple<torch::Tensor, torch::Tensor> local_weighted_average_backward(
    const torch::Tensor attention,
    const torch::Tensor values,
    const torch::Tensor grad
) {
    // Extract some shapes
    int N = attention.size(0);
    int H = attention.size(1);
    int L = attention.size(2);
    int local_context = attention.size(3);
    int S = values.size(2);
    int E = values.size(3);

    // Allocate space for the output
    auto grad_attention = torch::zeros_like(attention);
    auto grad_values = torch::zeros_like(values);

    // Create accessors for all the arguments
    auto aa = attention.accessor<float, 4>();
    auto va = values.accessor<float, 4>();
    auto ga = grad.accessor<float, 4>();
    auto gaa = grad_attention.accessor<float, 4>();
    auto gva = grad_values.accessor<float, 4>();

    #pragma omp parallel for collapse(2)
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            for (int l=0; l<L; l++) {
                int start = std::max(0, l-local_context/2);
                int end = std::min(S, l+(local_context+1)/2);
                int kstart = local_context/2 - std::abs(l-start);
                for (int s=start,k=kstart; s<end; k++, s++) {
                    scaled_copy_add(
                        &ga[n][h][l][0],
                        &gva[n][h][s][0],
                        aa[n][h][l][k],
                        E
                    );
                    gaa[n][h][l][k] = dot(
                        &ga[n][h][l][0],
                        &va[n][h][s][0],
                        E
                    );
                }
            }
        }
    }

    return std::make_tuple(grad_attention, grad_values);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "local_dot_product",
        &local_dot_product,
        "Compute the product of Q and K for a small context around each Q"
    );
    m.def(
        "local_dot_backward",
        &local_dot_backward,
        "Compute the gradient of local_dot_product"
    );
    m.def(
        "local_weighted_average",
        &local_weighted_average,
        "Perform the weighted average of V for a small context around each Q"
    );
    m.def(
        "local_weighted_average_backward",
        &local_weighted_average_backward,
        "Compute the gradient of the local weighted average"
    );
}

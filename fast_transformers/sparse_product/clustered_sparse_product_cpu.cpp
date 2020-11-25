//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <torch/extension.h>


inline float dot(float *a, float *b, int n) {
    float s = 0;
    for (int i=0; i<n; i++) {
        s += (*a) * (*b);
        a++;
        b++;
    }
    return s;
}


inline void add_scaled(float *a, float *b, float s, int n) {
    for (int i=0; i<n; i++) {
        (*a) += s * (*b);
        a++;
        b++;
    }
}


void clustered_sparse_dot_product(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor groups,
    const torch::Tensor topk,
    torch::Tensor product
) {
    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int k = topk.size(3);
    int S = K.size(2);
    int C = topk.size(2);

    float *queries = Q.data_ptr<float>();
    float *keys = K.data_ptr<float>();
    int64_t *topk_p = topk.data_ptr<int64_t>();
    int *groups_p = groups.data_ptr<int>();
    float *product_p = product.data_ptr<float>();

    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            for (int l=0; l<L; l++) {
                float *query = queries + n*H*L*E + h*L*E + l*E;
                int g = groups_p[n*H*L + h*L + l];
                if ((g > -1) && (g < C)) {
                    for (int j=0; j<k; j++) {
                        product_p[n*H*L*k + h*L*k + l*k + j] = dot(
                            query,
                            keys + n*H*S*E + h*S*E + topk_p[n*H*C*k + h*C*k + g*k + j]*E,
                            E
                        );
                    }
                }
            }
        }
    }
}


void clustered_sparse_dot_backward(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor groups,
    const torch::Tensor topk,
    const torch::Tensor grad_out,
    torch::Tensor grad_Q,
    torch::Tensor grad_K
) {
    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int k = topk.size(3);
    int S = K.size(2);
    int C = topk.size(2);

    float *queries = Q.data_ptr<float>();
    float *keys = K.data_ptr<float>();
    int64_t *topk_p = topk.data_ptr<int64_t>();
    int *groups_p = groups.data_ptr<int>();
    auto grad_out_a = grad_out.accessor<float, 4>();
    float *grad_q = grad_Q.data_ptr<float>();
    float *grad_k = grad_K.data_ptr<float>();

    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            for (int l=0; l<L; l++) {
                float *grad_query = grad_q + n*H*L*E + h*L*E + l*E;
                float *query = queries + n*H*L*E + h*L*E + l*E;
                int g = groups_p[n*H*L + h*L + l];
                if ((g > -1) && (g < C)) {
                    for (int j=0; j<k; j++) {
                        int key_idx = topk_p[n*H*C*k + h*C*k + g*k + j];
                        float g = grad_out_a[n][h][l][j];
                        add_scaled(
                            grad_query,
                            keys + n*H*S*E + h*S*E + key_idx*E,
                            g,
                            E
                        );
                        add_scaled(
                            grad_k + n*H*S*E + h*S*E + key_idx*E,
                            query,
                            g,
                            E
                        );
                    }
                }
            }
        }
    }
}


void clustered_sparse_weighted_average(
    const torch::Tensor weights,
    const torch::Tensor values,
    const torch::Tensor groups,
    const torch::Tensor topk,
    torch::Tensor output
) {
    int N = weights.size(0);
    int H = weights.size(1);
    int L = weights.size(2);
    int k = weights.size(3);
    int E = values.size(3);
    int C = topk.size(2);

    auto weights_a = weights.accessor<float, 4>();
    auto values_a = values.accessor<float, 4>();
    auto topk_a = topk.accessor<int64_t, 4>();
    auto groups_a = groups.accessor<int, 3>();
    auto output_a = output.accessor<float, 4>();

    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            for (int l=0; l<L; l++) {
                float *out = &output_a[n][h][l][0];
                int g = groups_a[n][h][l];
                if ((g > -1) && (g < C)) {
                    for (int j=0; j<k; j++) {
                        int key_idx = topk_a[n][h][g][j];
                        add_scaled(
                            out,
                            &values_a[n][h][key_idx][0],
                            weights_a[n][h][l][j],
                            E
                        );
                    }
                }
            }
        }
    }
}


void clustered_sparse_weighted_average_backward(
    const torch::Tensor weights,
    const torch::Tensor values,
    const torch::Tensor groups,
    const torch::Tensor topk,
    const torch::Tensor grad_out,
    torch::Tensor grad_weights,
    torch::Tensor grad_values
) {
    int N = weights.size(0);
    int H = weights.size(1);
    int L = weights.size(2);
    int k = weights.size(3);
    int E = values.size(3);
    int C = topk.size(2);

    auto weights_a = weights.accessor<float, 4>();
    auto values_a = values.accessor<float, 4>();
    auto topk_a = topk.accessor<int64_t, 4>();
    auto groups_a = groups.accessor<int, 3>();
    auto grad_out_a = grad_out.accessor<float, 4>();
    auto grad_weights_a = grad_weights.accessor<float, 4>();
    auto grad_values_a = grad_values.accessor<float, 4>();

    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            for (int l=0; l<L; l++) {
                float *grad = &grad_out_a[n][h][l][0];
                int g = groups_a[n][h][l];
                if ((g > -1) && (g < C)) {
                    for (int j=0; j<k; j++) {
                        int key_idx = topk_a[n][h][g][j];
                        add_scaled(
                            &grad_values_a[n][h][key_idx][0],
                            grad,
                            weights_a[n][h][l][j],
                            E
                        );
                        grad_weights_a[n][h][l][j] = dot(
                            &values_a[n][h][key_idx][0],
                            grad,
                            E
                        );
                    }
                }
            }
        }
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "clustered_sparse_dot_product",
        &clustered_sparse_dot_product,
        "Compute the dot product only in the positions specified by topk."
    );
    m.def(
        "clustered_sparse_dot_backward",
        &clustered_sparse_dot_backward,
        "Compute the gradients for the sparse dot product."
    );
    m.def(
        "clustered_sparse_weighted_average",
        &clustered_sparse_weighted_average,
        "Average the values using the sparse attention."
    );
    m.def(
        "clustered_sparse_weighted_average_backward",
        &clustered_sparse_weighted_average_backward,
        "Compute the gradients for the sparse weighted average."
    );
}

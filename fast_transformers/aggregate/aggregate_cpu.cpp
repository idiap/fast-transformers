//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <torch/extension.h>


/**
 * Aggregate the passed vectors X based on the group indices in G multiplied
 * by the factors F.
 */
void aggregate(
    const torch::Tensor X,
    const torch::Tensor G,
    const torch::Tensor F,
    torch::Tensor Y
) {
    int N = X.size(0);
    int H = X.size(1);
    int L = X.size(2);
    int E = X.size(3);

    int C = Y.size(2);
    const float *x = X.data_ptr<float>();
    const int32_t *g = G.data_ptr<int32_t>();
    const float *f = F.data_ptr<float>();
    float *y = Y.data_ptr<float>();

    // Aggregate all the Xs to the destination
    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            for (int l=0; l<L; l++) {
                int k = *(g + n*H*L + h*L + l);
                if ((k < 0) || (k >= C)) {
                    continue;
                }
                const float *src = x + n*H*L*E + h*L*E + l*E;
                float f_nhk = *(f + n*H*C + h*C + k);
                float *dst = y + n*H*C*E + h*C*E + k*E;

                for (int e=0; e<E; e++) {
                    *dst += (*src) * f_nhk;
                    dst++;
                    src++;
                }
            }
        }
    }
}


/**
 * Broadcast the aggregated results from tensor Y back to tensor X based on
 * group indices G multiplied by the factors F.
 */
void broadcast(
    const torch::Tensor Y,
    const torch::Tensor G,
    const torch::Tensor F,
    torch::Tensor X
) {
    int N = X.size(0);
    int H = X.size(1);
    int L = X.size(2);
    int E = X.size(3);

    int C = Y.size(2);

    const float *y = Y.data_ptr<float>();
    const int32_t *g = G.data_ptr<int32_t>();
    const float *f = F.data_ptr<float>();
    float *x = X.data_ptr<float>();

    // Broadcast all the Ys back into Xs
    // For now the parallelization is over L.
    // TODO: Check if parallelization over n is faster
    #pragma omp parallel for
    for (int l=0; l<L; l++) {
        for (int n=0; n<N; n++) {
            for (int h=0; h<H; h++) {
                int k = *(g + n*H*L + h*L + l);
                if ((k < 0) || (k >= C)) {
                    continue;
                }
                const float *src = y + n*H*C*E + h*C*E + k*E;
                float f_nhk = *(f + n*H*C + h*C + k);
                float *dst = x + n*H*L*E + h*L*E + l*E;

                for (int e=0; e<E; e++) {
                    *dst = *src * f_nhk;
                    dst++;
                    src++;
                }
            }
        }
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "aggregate",
        &aggregate,
        "Aggregate the vectors of X based on the "
        "indices in groups G multiplied by factors F."
    );
    m.def(
        "broadcast",
        &broadcast,
        "Broadcast the aggregated vectors Y back to X"
        "based on the indices in groups G multiplied by"
        "the factors F."
    );
}

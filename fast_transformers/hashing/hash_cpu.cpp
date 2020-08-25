//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <cassert>

#include <torch/extension.h>


/**
 * Hash the vectors in X with the hyperplanes A and store the result in H.
 * The positive side of the plane gets a 1 the negative a 0.
 */
void compute_hashes(torch::Tensor X, torch::Tensor A, torch::Tensor H) {
    float *x = X.data_ptr<float>();
    float *a = A.data_ptr<float>();
    int64_t *h = H.data_ptr<int64_t>();
    int N = X.size(0);
    int B = A.size(0);
    int D = X.size(1);
    assert(((void)"Bias expected for the parameters", D+1 == A.size(1)));
    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        int64_t hash = 0;
        for (int i=0; i<B; i++) {
            float s = 0;
            float * aij = a + i*(D+1);
            float * xnj = x + n*D;
            for (int j=0; j<D; j++) {
                s += (*xnj) * (*aij);
                xnj++;
                aij++;
            }
            hash |= (int64_t(s > (*aij))) << i;
        }
        h[n] = hash;
    }
}


/**
 * Hash the vectors given the projections on the B planes.
 * The positive side of the plane gets a 1 the negative a 0.
 */
void compute_hashes_from_projections(torch::Tensor P, torch::Tensor H) {
    float *p = P.data_ptr<float>();
    int64_t *h = H.data_ptr<int64_t>();
    int N = P.size(0);
    int B = P.size(1);
    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        int64_t hash = 0;
        float * pij = p + n*(B);
        for (int i=0; i<B; i++) {
            float s = *pij;
            pij++;
            hash |= (int64_t(s > 0)) << i;
        }
        h[n] = hash;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_hashes",
          &compute_hashes,
          "Hash the vectors X using SIMPLE-LSH.");
    m.def("compute_hashes_from_projections",
          &compute_hashes_from_projections,
          "Hash the vectors X given the computed projections.");
}

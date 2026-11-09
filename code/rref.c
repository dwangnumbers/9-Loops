/* In-place reduced row echelon form of a dense int64 matrix modulo a prime p < 2^31.
   Zero-skipping makes it fast on the very sparse matrices produced by the bootstrap.
   Build:  gcc -O3 -march=native -shared -fPIC -o librref.so rref.c            */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int64_t inv_mod(int64_t a, int64_t p) {
    int64_t t = 0, nt = 1, r = p, nr = a % p;
    if (nr < 0) nr += p;
    while (nr) { int64_t q = r / nr, tmp;
        tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp; }
    if (t < 0) t += p;
    return t;
}

/* returns rank; pivcols[0..rank) receives pivot column indices */
int rref_modp(int64_t *M, int rows, int cols, int64_t p, int *pivcols) {
    int rank = 0;
    int *nz = (int *)malloc(sizeof(int) * cols);
    for (int c = 0; c < cols && rank < rows; c++) {
        int piv = -1;
        for (int r = rank; r < rows; r++) if (M[(int64_t)r * cols + c]) { piv = r; break; }
        if (piv < 0) continue;
        if (piv != rank) {   /* swap rows */
            int64_t *A = M + (int64_t)piv * cols, *B = M + (int64_t)rank * cols;
            for (int k = 0; k < cols; k++) { int64_t t = A[k]; A[k] = B[k]; B[k] = t; }
        }
        int64_t *P = M + (int64_t)rank * cols;
        int64_t s = inv_mod(P[c], p);
        int nnz = 0;
        for (int k = c; k < cols; k++) if (P[k]) { P[k] = (P[k] * s) % p; nz[nnz++] = k; }
        for (int r = 0; r < rows; r++) {
            if (r == rank) continue;
            int64_t *R = M + (int64_t)r * cols;
            int64_t f = R[c];
            if (!f) continue;
            f = p - f;
            for (int t = 0; t < nnz; t++) { int k = nz[t]; R[k] = (R[k] + f * P[k]) % p; }
        }
        pivcols[rank++] = c;
    }
    free(nz);
    return rank;
}

/* In-place reduced row echelon form of a dense int64 matrix modulo a prime p < 2^31.
   Zero-skipping makes it fast on the very sparse matrices produced by the bootstrap.
   Build:  gcc -O3 -march=native -shared -fPIC -o librref.so rref.c            */
int rref_modp32(uint32_t *M, int rows, int cols, int64_t p, int *pivcols) {
    int rank = 0;
    int *nz = (int *)malloc(sizeof(int) * cols);
    for (int c = 0; c < cols && rank < rows; c++) {
        int piv = -1;
        for (int r = rank; r < rows; r++) if (M[(int64_t)r * cols + c]) { piv = r; break; }
        if (piv < 0) continue;
        if (piv != rank) {   /* swap rows */
            uint32_t *A = M + (int64_t)piv * cols, *B = M + (int64_t)rank * cols;
            for (int k = 0; k < cols; k++) { uint32_t t = A[k]; A[k] = B[k]; B[k] = t; }
        }
        uint32_t *P = M + (int64_t)rank * cols;
        int64_t s = inv_mod(P[c], p);
        int nnz = 0;
        for (int k = c; k < cols; k++) if (P[k]) { P[k] = (uint32_t)(((uint64_t)P[k] * (uint64_t)s) % (uint64_t)p); nz[nnz++] = k; }
        for (int r = 0; r < rows; r++) {
            if (r == rank) continue;
            uint32_t *R = M + (int64_t)r * cols;
            int64_t f = R[c];
            if (!f) continue;
            f = p - f;
            for (int t = 0; t < nnz; t++) { int k = nz[t]; R[k] = (uint32_t)(((uint64_t)R[k] + (uint64_t)f * (uint64_t)P[k]) % (uint64_t)p); }
        }
        pivcols[rank++] = c;
    }
    free(nz);
    return rank;
}

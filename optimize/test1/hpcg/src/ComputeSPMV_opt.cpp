#include "ComputeSPMV_opt.hpp"
#include <omp.h>
#include <immintrin.h> // For AVX intrinsics

int ComputeSPMV_opt(const SparseMatrix & A, Vector & x, Vector & y) {
    const local_int_t nrow = A.localNumberOfRows;
    double * const xv = x.values;
    double * const yv = y.values;
    
    // 修正指针类型
    double ** const matrixValues = A.matrixValues;
    local_int_t ** const mtxIndL = A.mtxIndL;
    double ** const matrixDiagonal = A.matrixDiagonal;

    // 预取数据到L1缓存
    #pragma omp parallel for
    for(local_int_t i=0; i<nrow; i++) {
        __builtin_prefetch(matrixValues[i], 0, 3);
        __builtin_prefetch(mtxIndL[i], 0, 3);
    }

    // 主计算循环
    #pragma omp parallel for schedule(dynamic,32)
    for(local_int_t i=0; i<nrow; i++) {
        const double * const cur_vals = matrixValues[i];
        const local_int_t * const cur_inds = mtxIndL[i];
        const local_int_t cur_nnz = A.nonzerosInRow[i];
        
        // 使用AVX向量化
        __m256d sum = _mm256_setzero_pd();
        local_int_t j = 0;
        
        // 4路展开的向量化循环
        for(; j+3 < cur_nnz; j+=4) {
            __m256d vals = _mm256_loadu_pd(&cur_vals[j]);
            __m256d xs = _mm256_set_pd(xv[cur_inds[j+3]], 
                                     xv[cur_inds[j+2]],
                                     xv[cur_inds[j+1]], 
                                     xv[cur_inds[j]]);
            sum = _mm256_fmadd_pd(vals, xs, sum);
        }
        
        // 处理剩余元素
        double temp = 0.0;
        for(; j < cur_nnz; j++) {
            temp += cur_vals[j] * xv[cur_inds[j]];
        }
        
        // 合并向量寄存器的结果
        double result[4];
        _mm256_storeu_pd(result, sum);
        yv[i] = result[0] + result[1] + result[2] + result[3] + temp;
    }

    return 0;
} 
#include <cstdio>
#include <cassert>
#include <iostream>

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include "ComputeDotProduct_opt.hpp"
#include "mytimer.hpp"

int ComputeDotProduct_opt(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

    assert(x.localLength >= n); 
    assert(y.localLength >= n);

    double local_result = 0.0;
    double * const xv = x.values;
    double * const yv = y.values;

    #pragma omp parallel for reduction(+:local_result)
    for (local_int_t i=0; i<n; i++) {
        local_result += xv[i] * yv[i];
    }

    // 使用MPI_Allreduce计算全局点积
    double t0 = mytimer();
#ifndef HPCG_NO_MPI
    double global_result = 0.0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    result = global_result;
#else
    result = local_result;
#endif
    time_allreduce = mytimer() - t0;

    isOptimized = true;
    return 0;
} 
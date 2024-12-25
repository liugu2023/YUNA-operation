// @HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
// @HEADER

/*!
// @file main.cpp

HPCG routine
*/

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "hpcg.hpp"
#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"

// CUDA 相关头文件
#include <cuda_runtime.h>  // CUDA运行时头文件
#include "spmv_cuda.cu"  // 引入CUDA优化的SPMV函数

/*! 
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.
  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.
  @return Returns zero on success and a non-zero value otherwise.
*/
int main(int argc, char * argv[]) {

#ifndef HPCG_NO_MPI
  MPI_Init(&argc, &argv);
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  // Check if QuickPath option is enabled.
  bool quickPath = (params.runningTime == 0);

  int size = params.comm_size, rank = params.comm_rank;  // Number of MPI processes, My process ID

  local_int_t nx, ny, nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;

  ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank == 0);
  if (ierr)
    return ierr;

  /////////////////////////
  // Problem setup Phase //
  /////////////////////////

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

  ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank == 0);
  if (ierr)
    return ierr;

  // Problem setup timer
  double setup_time = mytimer();

  SparseMatrix A;
  InitializeSparseMatrix(A, geom);

  Vector b, x, xexact;
  GenerateProblem(A, &b, &x, &xexact);
  SetupHalo(A);

  // Multi-level grid setup
  int numberOfMgLevels = 4;
  SparseMatrix * curLevelMatrix = &A;
  for (int level = 1; level < numberOfMgLevels; ++level) {
    GenerateCoarseProblem(*curLevelMatrix);
    curLevelMatrix = curLevelMatrix->Ac;
  }

  setup_time = mytimer() - setup_time;  // Capture total time of setup

  curLevelMatrix = &A;
  Vector * curb = &b;
  Vector * curx = &x;
  Vector * curxexact = &xexact;
  for (int level = 0; level < numberOfMgLevels; ++level) {
     CheckProblem(*curLevelMatrix, curb, curx, curxexact);
     curLevelMatrix = curLevelMatrix->Ac;  // Move to next coarse grid
     curb = 0;  // No vectors after the top level
     curx = 0;
     curxexact = 0;
  }

  CGData data;
  InitializeSparseCGData(A, data);

  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  // Record execution time of reference SpMV and MG kernels for reporting times
  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_overlap, b_computed;
  InitializeVector(x_overlap, ncol);
  InitializeVector(b_computed, nrow);

  // Load vector with random values
  FillRandomVector(x_overlap);

  int numberOfCalls = 10;
  if (quickPath) numberOfCalls = 1;  // QuickPath optimization

  double t_begin = mytimer();
  for (int i = 0; i < numberOfCalls; ++i) {
    ierr = ComputeSPMV_ref(A, x_overlap, b_computed);  // b_computed = A*x_overlap
    if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    ierr = ComputeMG_ref(A, b_computed, x_overlap);  // b_computed = Minv*y_overlap
    if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }

  times[8] = (mytimer() - t_begin) / ((double)numberOfCalls);  // Total time divided by number of calls.

  ///////////////////////////////
  // Reference CG Timing Phase //
  ///////////////////////////////

  // Compute the residual reduction for the natural ordering and reference kernels
  int niters = 0;
  int totalNiters_ref = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int refMaxIters = 50;
  numberOfCalls = 1;

  std::vector<double> ref_times(9, 0.0);
  double tolerance = 0.0;  // Set tolerance to zero to do maxIters iterations
  int err_count = 0;
  for (int i = 0; i < numberOfCalls; ++i) {
    ZeroVector(x);
    ierr = CG_ref(A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], true);
    if (ierr) ++err_count;
    totalNiters_ref += niters;
  }

  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
  double refTolerance = normr / normr0;

  /////////////////////////////////////
  // Optimize with CUDA for SPMV     //
  /////////////////////////////////////

  // Use CUDA to optimize SpMV (sparse matrix-vector multiplication)
  double t_cuda_begin = mytimer();

  // Assuming we have the device arrays ready and data transferred
  spmv_cuda(A.d_A, x_overlap.d_x, b_computed.d_b, A.d_indices, A.d_offsets, nrow);

  double t_cuda_end = mytimer();
  times[7] = t_cuda_end - t_cuda_begin;  // Capture CUDA optimization time

  ///////////////////////////////////
  // Final reporting and cleanup  //
  ///////////////////////////////////

  // 添加 TestNormsData 的实例
  TestNormsData testnorms_data;
  // 在这里调用相关的函数来填充 testnorms_data 的数据

  // 传递 testnorms_data
  ReportResults(A, 4, 1, 50, 10 * 50, &times[0], testcg_data, testsymmetry_data, testnorms_data, 0, quickPath);

#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}

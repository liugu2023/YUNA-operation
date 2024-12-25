#ifndef COMPUTEDOTPRODUCT_OPT_HPP
#define COMPUTEDOTPRODUCT_OPT_HPP

#include "Vector.hpp"

int ComputeDotProduct_opt(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized);

#endif 
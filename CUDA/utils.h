#ifndef UTILS_H
#define UTILS_H

/**
 * Generates a random double in the range of the input.
 */

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vector.h"

const double EPSILON = 1e-9; // std::numeric_limits<double>::epsilon()

double get_wall_time();

double get_cpu_time();

double rand_from(double, double);

__device__ __host__ bool isEqual(double, double);

__device__ __host__ double minDouble(double, double);

typedef struct
{
    double collisionTime;
    int collisionType;
} Pair;

#endif

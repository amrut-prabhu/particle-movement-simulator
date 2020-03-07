#ifndef SQUARE_H
#define SQUARE_H

#include "particle.h"
#include "utils.h" 
#include <limits>
#include <utility>

__device__ __host__ bool isOutOfBounds(Vector*, double, double);

__device__ __host__ bool hasExceededTopWall(Vector*, double, double);

__device__ __host__ bool hasExceededBottomWall(Vector*, double);

__device__ __host__ bool hasExceededRightWall(Vector*, double, double);

__device__ __host__ bool hasExceededLeftWall(Vector*, double);

__device__ __host__ Collision* identifyWallCollision(Particle*, double, double);

__device__ __host__ Pair identifyWallCollision(Vector* position, Vector* velocity, double squareSize,
                                                  double radius);

__device__ __host__ double getTimeForHorizontalWallCollision(Vector*, Vector*, double, double);

__device__ __host__ double getTimeForVerticalWallCollision(Vector*, Vector*, double, double);

__device__ __host__  double getTimeForOneDimWallCollision(double, double, double, double);

#endif

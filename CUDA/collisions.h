#ifndef COLLISIONS_H
#define COLLISIONS_H

#include <vector>
#include "particle.h"

__global__ void detectCollisions(int, int);

void sortAndFilterCollisions(std::vector<Collision>, int, Particle*);

__device__ Collision* getParticleCollision(Particle*, int, int, int);

__device__ Collision* getWallCollision(Particle*, double, double);

__host__ __device__ void printCollision(Collision);

#endif

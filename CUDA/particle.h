#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include "collision.h"
#include "vector.h"

typedef struct {
    int i; // the index of the particle from 0 to N-1
    int particleCollisions; // number of collisions with other particles
    int wallCollisions; // number of collisions with the walls
    Vector position; // position of particle
    Vector velocity; // velocity of particle
    Collision* currentCollision; // collision in current step (if any)
    // currentCollision will be == nullptr if there is no current collision
    Vector newPosition; // particle's position at start of next step
    Vector newVelocity; // particle's velocity at start of next step
} Particle;

/**
 * Returns a trasformed particle B in the frame of reference of particle A.
 */
Particle* transform(Particle*, Particle*);

__device__ __host__ Vector* getDestination(Particle*, double timeInterval);

double getTravelTime(Particle*, Vector* destination);

bool hasCollision(Particle*);

bool hasCollisionWithParticle(Particle*);

Vector* getConnectingVector(Particle*, Particle*);
#endif


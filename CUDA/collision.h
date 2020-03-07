#ifndef COLLISION_H
#define COLLISION_H

typedef struct {
    int A;
    int B; // index of particles that are involved in this collision (A < B)
    double collisionTime; // the time of collision
    // collisionTime will be between 0 and 1
    // it represents the amount of time after the start of the time step
} Collision;

Collision* deepCopy(Collision);

bool collisionComparator(Collision, Collision);

__host__ __device__ Collision makeNullCollision();

#endif

#include <stdlib.h>
#include "particle.h"

/**
 * Returns the trasformed particle B in the frame of reference of particle A.
 *
 * Given a particle A and B, express the velocity and position of particle B
 * from the frame of reference of particle A. Return a new Particle with these
 * properties.
 */
Particle* transform(Particle* A, Particle* B)
{

    Particle* transformedParticle = (Particle*)(malloc(sizeof(Particle)));
    transformedParticle->velocity.x = (B->velocity.x) - (A->velocity.x);
    transformedParticle->velocity.y = (B->velocity.y) - (A->velocity.y);
    transformedParticle->position.x = (B->position.x) - (A->position.x);
    transformedParticle->position.y = (B->position.y) - (A->position.y);

    return transformedParticle;
}

/**
 * Given a particle A, find the position of A after a certain time interval.
 * Return a Vector representing this destination position.
 *
 */
Vector* getDestination(Particle* A, double timeInterval)
{

    // calculate x coordinate of destination
    double x = (A->position.x) + (timeInterval * (A->velocity.x));
    // calculate y coordinate of destination
    double y = (A->position.y) + (timeInterval * (A->velocity.y));
    return make_vector(x, y);
}

/**
 * Given a destination point and a particle, find the time needed to reach
 * this destination.
 */
double getTravelTime(Particle* A, Vector* destination)
{
    // Find time spent travelling along x axis
    double xDistance = (A->position).x - (destination->x);
    double xTime = xDistance / ((A->velocity).x);

    // Find time spent travelling along y axis
    double yDistance = (A->position).y - (destination->y);
    double yTime = yDistance / ((A->velocity).y);

    // Sum to get total travel time
    return (xTime + yTime);
}

/**
 * Returns true iff the given particle has a collision.
 */
bool hasCollision(Particle* A)
{
    return (A->currentCollision) != nullptr;
}

/**
 * Returns true iff the given particle has a collision with another particle.
 */
bool hasCollisionWithParticle(Particle* A)
{
  return (A->currentCollision)->B >= 0;
}

/**
 * Given particles A and B, return the vector AB
 * i.e. the vector from centre of A to centre of B
 */
Vector* getConnectingVector(Particle* A, Particle* B)
{
    // Vector AB = Vector AO + Vector OB
    Vector* vector_OB = &(B->position);
    Vector* vector_OA = &(A->position);
    Vector* vector_AO = negate(vector_OA);
    Vector* vector_AB = add(vector_AO, vector_OB);

    free(vector_AO);
    return vector_AB;
}

#ifndef RESOLUTION_H
#define RESOLUTION_H

#include "square.h"

void resolveCollisions(Particle*, int, double, double);


/*
 * Given an array of particles, a collision and one of the particles in this
 * collision, return a pointer to the OTHER particle in this collision.
 */
Particle* getCollidingPartner(Particle*, Collision*, Particle*);


/*
 * Given a particle A and its colliding partner, find particle A's
 * velocity after the collision and update particle A.
 */
void resolveVelocity(Particle*, Particle*);

/** Given a particle A and its Collision, find A's position at the end
 *  of the current time interval, and update A.
 *
 *  Precondition: Particle A should already know its velocity after the
 *  collision.
 */
void resolvePosition(Particle*, Collision*, double, double);

void resolveVelocity(Particle*, Collision*);

#endif


#ifndef COLLISIONS_H
#define COLLISIONS_H

#include <vector>
#include "input.h"

std::vector<Collision> detectCollisions(Config*, Particle*);

void sortAndFilterCollisions(std::vector<Collision>, Config*, Particle*);

Collision* getParticleCollision(Particle*, int, int, int);

Collision* getWallCollision(Particle*, double, double);

#endif

#ifndef COLLISIONS_H
#define COLLISIONS_H

#include <vector>
#include "input.h"

std::vector<Collision> detectCollisions(Input*);

void sortAndFilterCollisions(std::vector<Collision>, Input*);

Collision* getParticleCollision(Particle*, int, int, int);

Collision* getWallCollision(Particle*, double, double);

#endif

#ifndef COLLISIONS_H
#define COLLISIONS_H

#include <vector>
#include "input.h"

std::vector<Collision> detectType1Collisions(Config*, Particle*, int);

std::vector<Collision> detectType2Collisions(Config*, Particle*, int, int);

void sortAndFilterCollisions(std::vector<Collision>, Config*, Particle*);

Collision* getParticleCollision(Particle*, int, int, int);

Collision* getWallCollision(Particle*, double, double);

#endif

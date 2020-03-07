#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vector>
#include "input.h"
#include "particle.h"
#include "watch.h"

void simulateStep(Input*, int, Watch*);

void printParticles(Particle*, int, int, bool a=false);

void updateParticles(Input*, int);

#endif

#ifndef INPUT_H
#define INPUT_H

#include "particle.h"
#include "utils.h"

typedef struct {
    int N; // Number of particles on the square surface
    int S; // Number of time steps to run the simulation for
    double L; // Size of the square
    double r; // Radius of the particles
    bool shouldPrint; // whether the position of each particle needs to be printed at each step
    int numPossibleCollisions;
    int chunkSize;
}  Config;

/**
 * Reads the input data from stdin.
 * If info about particles is not given, generates random values.
 */
void read_config(Config*);

void read_particles(Config*, Particle*);

/**
 * Prints the input data in the expected input format.
 */
void print_input(Config*, Particle*);

/**
 * Frees allocated memory.
 */
void cleanup_input(Config*, Particle*);

#endif

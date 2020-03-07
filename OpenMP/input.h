#ifndef INPUT_H
#define INPUT_H

#include "particle.h"
#include "utils.h"

typedef struct {
    int N; // Number of particles on the square surface
    double L; // Size of the square
    double r; // Radius of the particle
    int S; // Number of steps (time units) to run the simulation for
    bool shouldPrint; // whether the position of each particle needs to be printed at each step
    Particle* particles; // array of N particles
} Input;

/**
 * Reads the input data from stdin.
 * If info about particles is not given, generates random values.
 */
void read_input(Input*);

/**
 * Prints the input data in the expected input format.
 */
void print_input(Input*);

/**
 * Frees allocated memory.
 */
void cleanup_input(Input*);

#endif

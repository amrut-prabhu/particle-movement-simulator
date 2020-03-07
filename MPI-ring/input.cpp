#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "input.h"

/**
 * Reads the input data from stdin.
 * 
 * If info about particles is not given, generates random values:
 *  - Range for position along an axis: [r, L-r]
 *  - Range for velocity along an axis: [L/4, L/(8*r)]
 */
void read_config(Config* config)
{
    // SIMULATION PARAMETERS
    scanf("%d\n", &(config->N));
    scanf("%lf\n", &(config->L));
    scanf("%lf\n", &(config->r));
    scanf("%d\n", &(config->S));

    char outputType[6]; // max(len("print"), len("perf")) + 1
    scanf("%s\n", outputType);
    config->shouldPrint = !strcmp(outputType, "print");

}

void read_particles(Config* config, Particle* particles)
{
    // PARTICLES

    int idx;
    double x, y, vx, vy;

    int i = 0;
    for (; i < config->N; i++)
    {
        if (scanf("%d %lf %lf %lf %lf", &idx, &x, &y, &vx, &vy) == EOF)
        {
            break;
        }
        
        particles[idx].i = idx;
        particles[idx].position.x = x;
        particles[idx].position.y = y;
        particles[idx].velocity.x = vx;
        particles[idx].velocity.y = vy;
        particles[idx].particleCollisions = 0;
        particles[idx].wallCollisions = 0;
        particles[idx].currentCollision = nullptr;
    }

    if (i != 0) {
        return;
    }

    // Generate random positions and velocities for all particles
    srand (time(NULL));

    for (i = 0; i < config->N; i++)
    {
        particles[i].i = i;
        particles[i].position.x = rand_from(config->r, config->L - config->r);
        particles[i].position.y = rand_from(config->r, config->L - config->r);
        particles[i].velocity.x = rand_from(config->L / 4, config->L / (8 * config->r));
        particles[i].velocity.y = rand_from(config->L / 4, config->L / (8 * config->r));
        particles[i].particleCollisions = 0;
        particles[i].wallCollisions = 0;
    }
}

/**
 * Prints the input data in the expected input format.
 */
void print_input(Config* config, Particle* particles)
{
    printf("%d\n", config->N);
    printf("%10.8f\n", config->L);
    printf("%10.8f\n", config->r);
    printf("%d\n", config->S);
    printf("%s\n", config->shouldPrint ? "print" : "perf");

    Particle temp;
    for (int i = 0; i < config->N; i++)
    {
        temp = particles[i];
        printf("%d %10.8f %10.8f %10.8f %10.8f\n",
            temp.i, temp.position.x, temp.position.y, temp.velocity.x, temp.velocity.y);
    }

    printf("\n");
}

/**
 * Frees allocated memory.
 */
void cleanup_input(Config* config, Particle* particles)
{
    free(particles);
    free(config);
}

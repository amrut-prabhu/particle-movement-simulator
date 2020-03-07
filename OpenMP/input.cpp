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
void read_input(Input* input)
{
    // SIMULATION PARAMETERS
    scanf("%d\n", &(input->N));
    scanf("%lf\n", &(input->L));
    scanf("%lf\n", &(input->r));
    scanf("%d\n", &(input->S));

    char outputType[6]; // max(len("print"), len("perf")) + 1
    scanf("%s\n", outputType);
    input->shouldPrint= !strcmp(outputType, "print");

    // PARTICLES
    input->particles = (Particle*)malloc(sizeof(Particle) * input->N);

    int idx;
    double x, y, vx, vy;
 
    int i = 0;
    for (; i < input->N; i++)
    {
        if (scanf("%d %lf %lf %lf %lf", &idx, &x, &y, &vx, &vy) == EOF)
        {
            break;
        }
        
        input->particles[idx].i = idx;
        input->particles[idx].position.x = x;
        input->particles[idx].position.y = y;
        input->particles[idx].velocity.x = vx;
        input->particles[idx].velocity.y = vy;
        input->particles[idx].particleCollisions = 0;
        input->particles[idx].wallCollisions = 0;
        input->particles[idx].currentCollision = nullptr;
    }

    if (i != 0) {
        return;
    }

    // Generate random positions and velocities for all particles
    srand (time(NULL));

    for (i = 0; i < input->N; i++)
    {
        input->particles[i].i = i;
        input->particles[i].position.x = rand_from(input->r, input->L - input->r);
        input->particles[i].position.y = rand_from(input->r, input->L - input->r);
        input->particles[i].velocity.x = rand_from(input->L / 4, input->L / (8 * input->r));
        input->particles[i].velocity.y = rand_from(input->L / 4, input->L / (8 * input->r));
        input->particles[i].particleCollisions = 0;
        input->particles[i].wallCollisions = 0;
    }
}

/**
 * Prints the input data in the expected input format.
 */
void print_input(Input* input)
{
    printf("%d\n", input->N);
    printf("%10.8f\n", input->L);
    printf("%10.8f\n", input->r);
    printf("%d\n", input->S);
    printf("%s\n", input->shouldPrint ? "print" : "perf");

    Particle temp;
    for (int i = 0; i < input->N; i++)
    {
        temp = input->particles[i];
        printf("%d %10.8f %10.8f %10.8f %10.8f\n", 
            temp.i, temp.position.x, temp.position.y, temp.velocity.x, temp.velocity.y);
    }

    printf("\n");
}

/**
 * Frees allocated memory.
 */
void cleanup_input(Input* input)
{
    free(input->particles);
    free(input);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "collisions.h"
#include "simulator.h"
#include "resolution.h"

// #define MEASURE
// #define MEASURE_ALL // whether to measure wall clock time for each task
// #define CHECK // whether to do a sanity check on correctness

int NUM_THREADS = 40; // Number of threads used for parallelism for the different tasks

bool arePositionsNegative = false;

int main() 
{
    #ifdef MEASURE
        // Start timers
        Watch* watch = (Watch*)malloc(sizeof(Watch));
        watch->wall0 = get_wall_time();
        watch->cpu0  = get_cpu_time();
    #endif

    Input* input = (Input*)malloc(sizeof(Input));

    read_input(input);
    
    #ifdef MEASURE_ALL
        watch->setup_time += (get_wall_time() - watch->wall0);
    #endif

    printParticles(input->particles, input->N, 0);

    for (int i = 1; i <= input->S; i++) {

        #ifdef MEASURE_ALL
            simulateStep(input, NUM_THREADS, watch);
        #else
            simulateStep(input, NUM_THREADS, nullptr);
        #endif

        if (input->shouldPrint)
        {
            printParticles(input->particles, input->N, i, i == input->S);
        }
    }

    if (!input->shouldPrint)
    {
        printParticles(input->particles, input->N, input->S, true);
    }

    #ifdef MEASURE_ALL
        printf("Time for setup = %1.4f seconds\n", watch->setup_time);
        printf("Time for detecting collisions = %1.4f seconds\n", watch->collision_detection_time);
        printf("Time for sorting collisions = %1.4f seconds\n", watch->collision_sorting_time);
        printf("Time for resolving collisions = %1.4f seconds\n", watch->collision_resolution_time);
        printf("Time for updating particles = %1.4f seconds\n", watch->particle_update_time);
    #endif
    
    #ifdef MEASURE
        //  Stop timers
        double wall1 = get_wall_time();
        double cpu1  = get_cpu_time();

        printf("Wall Time = %1.4f seconds\n", wall1 - watch->wall0);
        printf("CPU Time = %1.4f seconds\n", cpu1 - watch->cpu0);
        
        free(watch);
        watch = nullptr;
    #endif
    
    #ifdef CHECK
        if (arePositionsNegative) printf("\n!!!===Some positions are invalid===!!!");
    #endif

    cleanup_input(input);
    return 0;
}

/**
 * Given particle velocities and positions at the start of a time step,
 * determine the velocities and positions at the end of this time step
 * - accounting for any collisions during the step.
 * 
 * Pass nullptr for the Watch* parameter if timing is not required.
 */
void simulateStep(Input* input, int NUM_THREADS, Watch* watch)
{
    // 1. Detect all collisions
    #ifdef MEASURE_ALL
        double time_before_detection = get_wall_time();
    #endif
    std::vector<Collision> possibleCollisions = detectCollisions(input, NUM_THREADS);
    #ifdef MEASURE_ALL
        watch->collision_detection_time += (get_wall_time() - time_before_detection);
    #endif

    // 2. Sort and filter the collisions 
    #ifdef MEASURE_ALL
        double time_before_sorting = get_wall_time();
    #endif
    sortAndFilterCollisions(possibleCollisions, input);
    #ifdef MEASURE_ALL
        watch->collision_sorting_time += (get_wall_time() - time_before_sorting);
    #endif

    // 3. Resolve updated positions and velocities after collisions 
    // find the velocities and positions of all particles at the end of the
    // current time step, accounting for any collisions
    #ifdef MEASURE_ALL
        double time_before_resolving = get_wall_time();
    #endif
    resolveCollisions(input, NUM_THREADS);
    #ifdef MEASURE_ALL
        watch->collision_resolution_time += (get_wall_time() - time_before_resolving);
    #endif

    // 4. Updated particle positions and velocities 
    #ifdef MEASURE_ALL
        double time_before_updating = get_wall_time();
    #endif
    // for each particle x, set velocity / position = newVelocity / newPosition
    updateParticles(input, NUM_THREADS);
    #ifdef MEASURE_ALL
        watch->particle_update_time += (get_wall_time() - time_before_updating);
    #endif
}

void printParticles(Particle* particles, int numParticles, int stepNum, bool hasSimulationEnded/*=false*/)
{
    for (int i = 0; i < numParticles; i++)
    {
        Particle* p = particles + i;     

        #ifdef CHECK
            arePositionsNegative = p->position.x < 0 || p->position.y < 0;
        #endif      

        if (hasSimulationEnded)
        {
            printf("%d %d %10.8f %10.8f %10.8f %10.8f %d %d\n", 
                stepNum, p->i, p->position.x, p->position.y, p->velocity.x, p->velocity.y, 
                p->particleCollisions, p->wallCollisions);
            continue;
        } 

        printf("%d %d %10.8f %10.8f %10.8f %10.8f\n", 
                stepNum, p->i, p->position.x, p->position.y, p->velocity.x, p->velocity.y);
    }
}

void updateParticles(Input* input, int NUM_THREADS)
{
    Particle* particles = input->particles;
    int numParticles = input->N;

    int i = 0;
    #pragma omp parallel num_threads(NUM_THREADS) shared(particles, numParticles) private(i)
    {
        #pragma omp for schedule(dynamic, 1)
        for (i = 0; i < numParticles; i++)
        {
            particles[i].velocity = particles[i].newVelocity;
            particles[i].position = particles[i].newPosition;
        }
    }

}

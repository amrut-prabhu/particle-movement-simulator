#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "collisions.h"
#include "simulator.h"
#include "resolution.h"

#define ROOT 0

int main(int argc, char** argv)
{
    int rank, worldSize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    Config* config = (Config*)malloc(sizeof(Config));
    Particle* particles;
    Particle* particlesPartition;
    Particle* partitionCopy;

    // Create the custom types
    MPI_Datatype MPI_COLLISION = create_collision_for_mpi();
    MPI_Datatype MPI_VECTOR = create_vector_for_mpi();
    MPI_Datatype MPI_PARTICLE = create_particle_for_mpi(MPI_VECTOR, MPI_COLLISION);
    MPI_Datatype MPI_CONFIG = create_config_for_mpi();

    if (rank == 0)
    {
        // master process will read input and perform initialisation
        read_config(config);
        particles = (Particle*)malloc(sizeof(Particle)*(config->N));
        read_particles(config, particles);
        printParticles(particles, config->N, 0);
    }

    // send config to all slaves
    MPI_Bcast(config, 1, MPI_CONFIG, ROOT, MPI_COMM_WORLD);

    config->numPossibleCollisions = ((config->N) * ((config->N) + 1)) / 2;
    int numSlaves = worldSize - 1;
    config->chunkSize = (config->N) / worldSize;

    // all slaves need a buffer for their original partition
    particlesPartition = (Particle*)malloc(sizeof(Particle)*(config->chunkSize));

    // all slaves need a buffer for their working partition
    partitionCopy = (Particle*)malloc(sizeof(Particle)*(config->chunkSize));

    // run the simulation
    for (int i = 1; i <= config->S; i++)
    {
        // send buffer of particles to slaves
        MPI_Scatter(particles, config->chunkSize, MPI_PARTICLE, particlesPartition, config->chunkSize, MPI_PARTICLE, ROOT, MPI_COMM_WORLD);
        if (rank == 0)
        {

            std::vector<Collision> masterCollisions = slaveSimulateStep(config, particlesPartition, partitionCopy, ROOT, worldSize, 
                                                                            MPI_PARTICLE, MPI_COLLISION);
            masterSimulateStep(config, particles, masterCollisions, MPI_PARTICLE, MPI_COLLISION);

            if (config->shouldPrint)
            {
                printParticles(particles, config->N, i, i == config->S);
            }

        } else
        {
            slaveSimulateStep(config, particlesPartition, partitionCopy, rank, worldSize, MPI_PARTICLE, MPI_COLLISION);
        }

    }

    if (rank == 0)
    {
      if (!config->shouldPrint)
      {
          printParticles(particles, config->N, config->S, true);
      }
      cleanup_input(config, particles);
    }

    MPI_Finalize();
    return 0;
}

/* Simulates a single time step in the master process */
void masterSimulateStep(Config* config, Particle* particles, std::vector<Collision> possibleCollisions, 
                            MPI_Datatype MPI_PARTICLE, MPI_Datatype MPI_COLLISION) {

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // receive detected collisions from slaves
    std::vector<Collision> allPossibleCollisions = possibleCollisions;
    for (int j = 1; j < worldSize; j++)
    {
        Collision* receivedCollisions = (Collision*)malloc(sizeof(Collision)*(config->numPossibleCollisions));
        MPI_Status status;
        int numCollisionsReceived;
        int tag = 0; // TODO: The tag value seems meaningless in our program?

        MPI_Recv(receivedCollisions, config->numPossibleCollisions, MPI_COLLISION, j, tag, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_COLLISION, &numCollisionsReceived);
        allPossibleCollisions.insert(allPossibleCollisions.end(), receivedCollisions, receivedCollisions + numCollisionsReceived);
        free(receivedCollisions);
    }
    // sort and filter collisions
    sortAndFilterCollisions(allPossibleCollisions, config, particles);
    // resolve collisions
    resolveCollisions(config, particles);

    // update particle velocities and positions
    updateParticles(config, particles);
}


/* Simulates a single time step for each slave process */
std::vector<Collision> slaveSimulateStep(Config* config, Particle* particles, Particle* particlesCopy, int rank, int numOfStages, 
                                                  MPI_Datatype MPI_PARTICLE, MPI_Datatype MPI_COLLISION)
{

    Particle* receiveBuffer = (Particle*)malloc(sizeof(Particle)*config->chunkSize);

    // TODO: what tag to use?
    int tag = 0;
    int nextStageIndex = (rank + 1) >= numOfStages ? 0 : rank + 1;
    int previousStageIndex = (rank - 1) < 0 ? numOfStages - 1 : rank - 1;
    std::vector<Collision> possibleCollisions;
    std::vector<Collision> stageCollisions;

    // copy the received partition into its working partition
    for (int i = 0; i < config->chunkSize; i++)
    {
        particlesCopy[i] = particles[i];
    }

    for (int currentStage = 0; currentStage < numOfStages; currentStage++)
    {
        if (currentStage <= rank)
        {
            stageCollisions = detectCollisions(config, particlesCopy, particles);
        } else
        {
            stageCollisions = detectCollisions(config, particles, particlesCopy);
        }

        possibleCollisions.insert(possibleCollisions.end(), stageCollisions.begin(), stageCollisions.end());

        // send particles' partition to next stage in pipeline
        MPI_Request request;
        MPI_Isend(particlesCopy, config->chunkSize, MPI_PARTICLE, nextStageIndex, tag, MPI_COMM_WORLD, &request);

        // get particles' partition from previous stage in pipeline
        MPI_Recv(receiveBuffer, config->chunkSize, MPI_PARTICLE, previousStageIndex, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

        MPI_Wait(&request, MPI_STATUS_IGNORE);
       
        for(int i = 0; i < config->chunkSize; i++)
        {
            particlesCopy[i] = receiveBuffer[i];
        }
    
    }
    
    if (rank != 0)  
    {
        // return detected collisions to master
        Collision* collisionsBuffer = possibleCollisions.data();
        MPI_Send(collisionsBuffer, possibleCollisions.size(), MPI_COLLISION, ROOT, tag, MPI_COMM_WORLD);
    }

    free(receiveBuffer);
    return possibleCollisions;
}

void printParticles(Particle* particles, int numParticles, int stepNum, bool hasSimulationEnded/*=false*/)
{
    for (int i = 0; i < numParticles; i++)
    {
        Particle* p = particles + i;

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

void updateParticles(Config* config, Particle* particles)
{
    int numParticles = config->N;

    for (int i = 0; i < numParticles; i++)
    {
        particles[i].velocity = particles[i].newVelocity;
        particles[i].position = particles[i].newPosition;
    }
}


MPI_Datatype create_collision_for_mpi()
{
    MPI_Datatype MPI_COLLISION;
    int numBlocks = 2;
    int blockLengths[numBlocks] = { 2, 1 };
    const MPI_Aint displacements[numBlocks] = { 0, sizeof(int)*2 };
    MPI_Datatype oldTypes[numBlocks] = { MPI_INT, MPI_DOUBLE };
    MPI_Type_create_struct(numBlocks, blockLengths, displacements, oldTypes, &MPI_COLLISION);
    MPI_Type_commit(&MPI_COLLISION);

    return MPI_COLLISION;
}


MPI_Datatype create_vector_for_mpi()
{
    MPI_Datatype MPI_VECTOR;
    int numBlocks = 1;
    int blockLengths[numBlocks] = { 2 };
    const MPI_Aint displacements[numBlocks] = { 0 };
    MPI_Datatype oldTypes[numBlocks] = { MPI_DOUBLE };
    MPI_Type_create_struct(numBlocks, blockLengths, displacements, oldTypes, &MPI_VECTOR);
    MPI_Type_commit(&MPI_VECTOR);

    return MPI_VECTOR;
}

MPI_Datatype create_particle_for_mpi(MPI_Datatype MPI_VECTOR, MPI_Datatype MPI_COLLISION)
{
    MPI_Datatype MPI_PARTICLE;
    int numBlocks = 3;
    int blockLengths[numBlocks] = { 2, 4, 1 };
    const MPI_Aint displacements[numBlocks] = { 0, sizeof(int)*2, sizeof(int)*2 + sizeof(Vector)*4 };
    MPI_Datatype oldTypes[numBlocks] = { MPI_DOUBLE, MPI_VECTOR, MPI_COLLISION };
    MPI_Type_create_struct(numBlocks, blockLengths, displacements, oldTypes, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);

    return MPI_PARTICLE;
}

MPI_Datatype create_config_for_mpi()
{
    MPI_Datatype MPI_CONFIG;
    int numBlocks = 2;
    int blockLengths[numBlocks] = { 2, 2 };
    const MPI_Aint displacements[numBlocks]
        = { 0, sizeof(int)*2 };
    MPI_Datatype oldTypes[numBlocks] = { MPI_INT, MPI_DOUBLE };
    MPI_Type_create_struct(numBlocks, blockLengths, displacements, oldTypes, &MPI_CONFIG);
    MPI_Type_commit(&MPI_CONFIG);

    return MPI_CONFIG;
}



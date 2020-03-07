#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include <set> 

#include "collisions.h"
#include "simulator.h"
#include "resolution.h"

using namespace std; 

// #define USEDEBUG


int tag = 1; // TODO: The tag value seems meaningless in our program?
int numProcesses;

#define M 4 // Number of partitions to break the particles array into

// TODO: convert to int-s instead

// Rank of the root master process
#define MASTER 0 //(numProcesses - 1)

// Ranks of slaves
#define SLAVE_START 1
#define SLAVE_END (M * (M + 1) / 2)

// Ranks of sub-master processes on each node
#define SUB_MASTER_START (SLAVE_END + 1)
#define SUB_MASTER_END (numProcesses - 1)

int *slaveTypes; // Types of processes SLAVE_START to SLAVE_END, i.e., Type 1 or Type 2 slave

int main(int argc, char** argv)
{
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    generateSlaveTypes();

    Config* config = (Config*)malloc(sizeof(Config));
    Particle* particles;

    // Create the custom types
    MPI_Datatype MPI_COLLISION = create_collision_for_mpi();
    MPI_Datatype MPI_VECTOR = create_vector_for_mpi();
    MPI_Datatype MPI_PARTICLE = create_particle_for_mpi(MPI_VECTOR, MPI_COLLISION);
    MPI_Datatype MPI_CONFIG = create_config_for_mpi();

    if (rank == MASTER)
    {
        // master process will read input and perform initialisation
        read_config(config);
        particles = (Particle*)malloc(sizeof(Particle)*(config->N));
        read_particles(config, particles);
        printParticles(particles, config->N, 0);
    
        int partitionSize = ceil(config->N * 1.0 / M);
        config->numPossibleCollisions = max(partitionSize * (partitionSize + 1) / 2, partitionSize * partitionSize);

        int numSlaves = numProcesses - 1;
        config->chunkSize = ceil((double)(config->numPossibleCollisions) / (double)numSlaves);
    }
    
    // send config to all other processes
    MPI_Bcast(config, 1, MPI_CONFIG, MASTER, MPI_COMM_WORLD);

    map<int, map<int, vector<int>>> allocations = allocatePartitions(rank);

    map<int, set<int>> submasterSlaves; 
    map<int, int> slaveSubmaster; 
    for (int submaster = SUB_MASTER_START; submaster <= SUB_MASTER_END; submaster++) {
        set<int> slaves;

        for(auto partitionAllocation: allocations[submaster]) {
            vector<int> currSlaves = partitionAllocation.second;
            for(int slave: currSlaves) {
                slaves.insert(slave); 
                slaveSubmaster.insert(make_pair(slave, submaster));
            }
            // std::copy(currSlaves.begin(), currSlaves.end(), std::inserter(slaves, slaves.end()));
        }

        submasterSlaves.insert(make_pair(submaster, slaves));
    }

    #ifdef USEDEBUG
    if (rank == MASTER) {
        printAllocations(allocations);
    }
    #endif

    int partitionSize = ceil(config->N * 1.0 / M); // TODO: add to config (instead of chunkSize)

    // Allocate memory for particles in non-master processes
    if (isSlaveType1(rank)) {   
        particles = (Particle*)malloc(sizeof(Particle)*(partitionSize));
    } else if (isSlaveType2(rank)) {   
        particles = (Particle*)malloc(sizeof(Particle)*(2 * partitionSize));
    } else if (isSubmaster(rank)) {
        int numPartitions = allocations[rank].size();
        particles = (Particle*)malloc(sizeof(Particle)*(numPartitions * partitionSize));
    }

    // run the simulation
    for (int i = 1; i <= config->S; i++)
    {
        if (rank == MASTER) {
            masterSimulateStep(config, particles, allocations, submasterSlaves, MPI_PARTICLE, MPI_COLLISION);
            if (config->shouldPrint) {
                printParticles(particles, config->N, i, i == config->S);
            }
        } else if (isSubmaster(rank)) {
            submasterSimulateStep(config, particles, allocations[rank], submasterSlaves[rank], MPI_PARTICLE, MPI_COLLISION);
        } else if (isSlaveType1(rank)) {   
            slaveType1SimulateStep(config, particles, slaveSubmaster[rank], MPI_PARTICLE, MPI_COLLISION);
        } else if (isSlaveType2(rank)) {
            slaveType2SimulateStep(config, particles, slaveSubmaster[rank], MPI_PARTICLE, MPI_COLLISION);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        #ifdef USEDEBUG
        printf("===========================================Step %d\n", i);
        #endif
    }

    if (rank == MASTER) {
        if (!config->shouldPrint) {
            printParticles(particles, config->N, config->S, true);
        }
    }

    cleanup_input(config, particles);
    free(slaveTypes);

    MPI_Finalize();
    return 0;
}

void generateSlaveTypes()
{
    int numSlaves = SLAVE_END - SLAVE_START + 1;
    slaveTypes = (int*)malloc(sizeof(int) * numSlaves);

    for (int slave = SLAVE_START; slave <= SLAVE_END; slave++) 
    {
        pair<int, int> partitions = getRequiredPartitions(slave);
        if (partitions.first == partitions.second) {
            // #ifdef USEDEBUG
            // printf("Slave %d is Type1\n", slave);
            // #endif
            slaveTypes[slave - 1] = 1;            
        } else {
            // #ifdef USEDEBUG
            // printf("Slave %d is Type2\n", slave);
            // #endif
            slaveTypes[slave - 1] = 2;            
        }
    }
}

bool isSlaveType1(int rank) { return slaveTypes[rank - 1] == 1; }
bool isSlaveType2(int rank) { return slaveTypes[rank - 1] == 2; }
bool isSubmaster(int rank) { return SUB_MASTER_START <= rank && rank <= SUB_MASTER_END; }

/**
 * Simulates a single time step in the master process
 */
void masterSimulateStep(Config* config, Particle* particles, 
                        map<int, map<int, vector<int>>> allocations, map<int, set<int>> submasterSlaves,
                        MPI_Datatype MPI_PARTICLE, MPI_Datatype MPI_COLLISION) 
{
    int partitionSize = ceil(config->N * 1.0 / M);
    
    // send buffer of particles to submasters
    for (int submaster = SUB_MASTER_START; submaster <= SUB_MASTER_END; submaster++) 
    {
        map<int, vector<int>> requiredPartitions = allocations[submaster];

        for(auto partitionAllocation : requiredPartitions) {
            int partitionIdx = partitionAllocation.first;
            int startIdx = partitionIdx * partitionSize;
            int size = (partitionIdx == M - 1) ? config->N - (partitionSize * (M - 1)) : partitionSize; // FIXME: small N (eg. 1) gives -ve 

            MPI_Request request;
            #ifdef USEDEBUG
            printf("Master sending partition %d to submaster %d with size=%d and start=%d\n", partitionIdx, submaster, size, startIdx);
            #endif
            MPI_Isend(particles + startIdx, size, MPI_PARTICLE, submaster, partitionIdx, MPI_COMM_WORLD, &request); // TODO: check edge cases
            // MPI_Send(particles + startIdx, size, MPI_PARTICLE, submaster, partitionIdx, MPI_COMM_WORLD); // TODO: check edge cases
        }
    }

    // receive detected collisions from submasters
    int numRequests = SLAVE_END - SLAVE_START + 1;
    MPI_Request requests[numRequests];
    MPI_Status statuses[numRequests];

    std::vector<Collision> allPossibleCollisions;
    int idx = 0;
    for (int submaster = SUB_MASTER_START; submaster <= SUB_MASTER_END; submaster++) {
        for (auto slave: submasterSlaves[submaster]) {
            int numPossibleCollisions = isSlaveType1(slave) ? partitionSize * (partitionSize + 1) / 2 : partitionSize * partitionSize; // TODO: avoid recomputation
            
            Collision* receivedCollisions = (Collision*)malloc(sizeof(Collision)*(numPossibleCollisions)); // TODO: move numPossibleCollisions to config
            int numCollisionsReceived;

            // MPI_Irecv(receivedCollisions, numPossibleCollisions, MPI_COLLISION, submaster, slave, MPI_COMM_WORLD, &requests[idx]);
            MPI_Recv(receivedCollisions, numPossibleCollisions, MPI_COLLISION, submaster, slave, MPI_COMM_WORLD, &statuses[idx]);
            MPI_Get_count(&statuses[idx], MPI_COLLISION, &numCollisionsReceived);
            #ifdef USEDEBUG
            printf("Master received %d collisions from submaster %d - slave %d\n", numCollisionsReceived, submaster, slave);
            #endif

            allPossibleCollisions.insert(allPossibleCollisions.end(), receivedCollisions, receivedCollisions + numCollisionsReceived);
            free(receivedCollisions);
            idx++;
        }
    }
    // MPI_Waitall(numRequests, requests, statuses);
    #ifdef USEDEBUG
    printf("Master received collisions from %d submaster-slaves\n", numRequests);
    #endif


    // sort and filter collisions
    sortAndFilterCollisions(allPossibleCollisions, config, particles);

    // resolve collisions
    resolveCollisions(config, particles);

    // update particle velocities and positions
    updateParticles(config, particles);
}

/**
 * Simulates a single time step for each submaster process
 */
void submasterSimulateStep(Config* config, Particle* particles, 
                            map<int, vector<int>> partitionAllocations, set<int> slaves,
                            MPI_Datatype MPI_PARTICLE, MPI_Datatype MPI_COLLISION)
{
    int partitionSize = ceil(config->N * 1.0 / M);

    // Receive particles partitions from master
        int numPartitions = partitionAllocations.size();
    Particle partitions[numPartitions][partitionSize];
    int tags[numPartitions];
    MPI_Request requests[numPartitions];
    MPI_Status statuses[numPartitions];

    // for(int i = 0; i < numPartitions; i++) {
    int i = 0;
    for (auto allocation: partitionAllocations) {
        int partitionIdx = allocation.first;

        MPI_Irecv(&partitions[i], partitionSize, MPI_PARTICLE, MASTER, partitionIdx, MPI_COMM_WORLD, &requests[i]);

        tags[i] = partitionIdx;
        i++;
    }
    MPI_Waitall(numPartitions, requests, statuses);
    #ifdef USEDEBUG
    printf("  Submaster received partitions %d and %d\n", tags[0], tags[1]);
    #endif


    // Send partitions to slaves of this submaster (on the same node)
    for (int i = 0; i < numPartitions; i++) 
    {
        int partitionIdx = tags[i];
        vector<int> currSlaves = partitionAllocations[partitionIdx];

        for(int slave: currSlaves) {
            int startIdx = partitionIdx * partitionSize;
            int size = (partitionIdx == M - 1) ? config->N - (partitionSize * (M - 1)) : partitionSize;
            // int size; // = (partitionIdx == M - 1) ? config->N - (partitionSize * (M - 1)) : partitionSize;
            // MPI_Get_count(&statuses[i], MPI_COLLISION, &size);
            #ifdef USEDEBUG
            printf("  Submaster sending partition %d to slave %d with size=%d\n", partitionIdx, slave, size);
            #endif

            MPI_Request request;
            MPI_Isend(&partitions[i], size, MPI_PARTICLE, slave, tag, MPI_COMM_WORLD, &request);
        }
    }

    // Receive detected collisions from slaves 
    // and Send to master
    int numCollRequests = slaves.size();
    MPI_Status collStatuses[numCollRequests];
    // MPI_Request collRequests[numCollRequests];

    int idx = 0;
    for (auto slave: slaves) {
        int numPossibleCollisions = isSlaveType1(slave) ? partitionSize * (partitionSize + 1) / 2 : partitionSize * partitionSize;
        
        Collision* receivedCollisions = (Collision*)malloc(sizeof(Collision)*(numPossibleCollisions));
        int numCollisionsReceived;

        MPI_Recv(receivedCollisions, numPossibleCollisions, MPI_COLLISION, slave, tag, MPI_COMM_WORLD, &collStatuses[idx]);
        MPI_Get_count(&collStatuses[idx], MPI_COLLISION, &numCollisionsReceived);
        #ifdef USEDEBUG
        printf("  Submaster received %d/%d collisions from slave %d\n", numCollisionsReceived, numPossibleCollisions, slave);
        #endif

        #ifdef USEDEBUG
        printf("  Submaster sending %d collisions from slave %d to master\n", numCollisionsReceived, slave);
        #endif
        MPI_Send(receivedCollisions, numCollisionsReceived, MPI_COLLISION, MASTER, slave, MPI_COMM_WORLD);

        free(receivedCollisions);
        idx++;
    }
    // MPI_Waitall(numCollRequests, collRequests, collStatuses);
}


/**
 * Simulates a single time step for each Type 1 slave process (detects collisions within 1 partition).
 */
void slaveType1SimulateStep(Config* config, Particle* particles, int submaster, MPI_Datatype MPI_PARTICLE, MPI_Datatype MPI_COLLISION)
{
    int partitionSize = ceil(config->N * 1.0 / M);
    int numParticles;
    
    // get buffer of particles from submaster
    MPI_Status status;
    MPI_Recv(particles, partitionSize, MPI_PARTICLE, submaster, tag, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_PARTICLE, &numParticles);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef USEDEBUG
    printf("    Slave %d received partition with size %d\n", rank, numParticles);
    #endif


    // detect collisions
    std::vector<Collision> possibleCollisions = detectType1Collisions(config, particles, numParticles); // TODO:

    // return detected collisions to submaster
    Collision* collisionsBuffer = possibleCollisions.data();
    #ifdef USEDEBUG
    printf("    Slave %d sending %d collisions1 to submaster %d\n", rank, possibleCollisions.size(), submaster);
    #endif
    MPI_Send(collisionsBuffer, possibleCollisions.size(), MPI_COLLISION, submaster, tag, MPI_COMM_WORLD);
}

/** 
 * Simulates a single time step for each Type 2 slave process (detects collisions across 2 partitions).
 */
void slaveType2SimulateStep(Config* config, Particle* particles, int submaster, MPI_Datatype MPI_PARTICLE, MPI_Datatype MPI_COLLISION)
{
    int partitionSize = ceil(config->N * 1.0 / M);
    int numParticles1, numParticles2;

    // get buffer of particles from submaster
    MPI_Status status;
    MPI_Recv(particles, partitionSize, MPI_PARTICLE, submaster, tag, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_PARTICLE, &numParticles1);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef USEDEBUG
    printf("    Slave %d received partition1 with size %d\n", rank, numParticles1);
    #endif

    MPI_Recv(particles + numParticles1, partitionSize, MPI_PARTICLE, submaster, tag, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_PARTICLE, &numParticles2);
    #ifdef USEDEBUG
    printf("    Slave %d received partition2 with size %d\n", rank, numParticles2);
    #endif

    std::vector<Collision> possibleCollisions = detectType2Collisions(config, particles, numParticles1, numParticles2);  // TODO:

    // return detected collisions to submaster
    Collision* collisionsBuffer = possibleCollisions.data();
    #ifdef USEDEBUG
    printf("    Slave %d sending %d collisions2 to submaster %d\n", rank, possibleCollisions.size(), submaster);
    #endif
    MPI_Send(collisionsBuffer, possibleCollisions.size(), MPI_COLLISION, submaster, tag, MPI_COMM_WORLD);
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


/**
 * Returns 
 * { 
 *   submaster_rank : { 
 *     partition_idx: [ slave_rank ] 
 *   }
 * }
 */
map<int, map<int, vector<int>>> allocatePartitions(int rank)
{
    char hosts[numProcesses][1024];
    gethostname(hosts[rank], 1024);

	for (int i = 0; i < numProcesses; i++) {
    	MPI_Bcast(&hosts[i], 1024, MPI_CHAR, i, MPI_COMM_WORLD);
    }

    
    // Generate allocations
    map<int, map<int, vector<int>>> allocations;

    for (int submaster = SUB_MASTER_START; submaster <= SUB_MASTER_END; submaster++) {
        map<int, vector<int>> allocation;  // { partition_idx: [slaves on submaster's node that need this partition] }

        for (int slave = SLAVE_START; slave <= SLAVE_END; slave++) {
            if (strcmp(hosts[submaster], hosts[slave]) == 0) {
                pair<int, int> partitions = getRequiredPartitions(slave);

                insertAllocation(allocation, partitions.first, slave);

                if (partitions.first != partitions.second) {
                    insertAllocation(allocation, partitions.second, slave);
                }
            }
        }

        allocations.insert(make_pair(submaster, allocation));
    }

    return allocations;
}

pair<int, int> getRequiredPartitions(int k) {
    int idx = k - 1; // Need to offset because rank 0 is taken by MASTER

    int i = idx / (M + 1);
    int j = idx % (M + 1);

    if (j > i)
    {
        i = M - i - 1;
        j = M - j;
    }

    return make_pair(i, j);
}

void insertAllocation(map<int, vector<int>> &map, int key, int element)
{
    if (map.count(key) == 0) {
        map.insert(make_pair(key, vector<int>()));
    }
    map[key].push_back(element);
}

void printAllocations(map<int, map<int, vector<int>>> &allocations)
{
    // Paritions needed by each slave
    for (int slave = SLAVE_START; slave <= SLAVE_END; slave++) {
        pair<int, int> partitions = getRequiredPartitions(slave);
        if (partitions.first == partitions.second) {
            printf("Slave %d : %c (%d)\n", slave, 65 + partitions.first, partitions.first);
        } else {
            printf("Slave %d : %c%c (%d, %d)\n", slave, 65 + partitions.first, 65 + partitions.second, partitions.first, partitions.second);
        }
    }

    // Allocations for submasters
    printf("{\n ");
    for(auto elem : allocations) {
        printf("  submaster %d : {\n", elem.first);
        
        for(auto subelem : elem.second) {
            printf("    Partition %c : slaves [", subelem.first + 65);
            std::copy(subelem.second.begin(), subelem.second.end(), std::ostream_iterator<int>(std::cout, " "));
            printf("]\n");
        }
        printf("  }\n");
    }
    printf("}\n");
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
    int numBlocks = 3;
    int blockLengths[numBlocks] = { 4, 2, 1 };
    const MPI_Aint displacements[numBlocks]
        = { 0, sizeof(int)*4, sizeof(int)*4 + sizeof(double)*2 };
    MPI_Datatype oldTypes[numBlocks] = { MPI_INT, MPI_DOUBLE, MPI_INT };
    MPI_Type_create_struct(numBlocks, blockLengths, displacements, oldTypes, &MPI_CONFIG);
    MPI_Type_commit(&MPI_CONFIG);

    return MPI_CONFIG;
}



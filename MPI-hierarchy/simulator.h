#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <map> 
#include <utility>
#include <vector>
#include <iterator> 

#include "input.h"
#include "particle.h"
#include "watch.h"

void generateSlaveTypes();

bool isSlaveType1(int);

bool isSlaveType2(int);

bool isSubmaster(int);



std::map<int, std::map<int, std::vector<int>>> allocatePartitions(int);

std::pair<int, int> getRequiredPartitions(int); 

void insertAllocation(std::map<int, std::vector<int>> &map, int, int);

void printAllocations(std::map<int, std::map<int, std::vector<int>>> &allocations);



void masterSimulateStep(Config*, Particle*, std::map<int, std::map<int, std::vector<int>>>, std::map<int, std::set<int>>, MPI_Datatype, MPI_Datatype);

void submasterSimulateStep(Config*, Particle*, std::map<int, std::vector<int>>, std::set<int>, MPI_Datatype, MPI_Datatype);

void slaveType1SimulateStep(Config*, Particle*, int, MPI_Datatype, MPI_Datatype);

void slaveType2SimulateStep(Config*, Particle*, int, MPI_Datatype, MPI_Datatype);



void printParticles(Particle*, int, int, bool a=false);

void updateParticles(Config*, Particle*);

MPI_Datatype create_collision_for_mpi();

MPI_Datatype create_vector_for_mpi();

MPI_Datatype create_particle_for_mpi(MPI_Datatype, MPI_Datatype);

MPI_Datatype create_config_for_mpi();

#endif

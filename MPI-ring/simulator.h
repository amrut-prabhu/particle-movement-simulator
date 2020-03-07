#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vector>
#include "input.h"
#include "particle.h"
#include "watch.h"

void masterSimulateStep(Config*, Particle*, std::vector<Collision>, MPI_Datatype, MPI_Datatype);

std::vector<Collision> slaveSimulateStep(Config*, Particle*, Particle*, int, int, MPI_Datatype, MPI_Datatype);

void printParticles(Particle*, int, int, bool a=false);

void updateParticles(Config*, Particle*);

MPI_Datatype create_collision_for_mpi();

MPI_Datatype create_vector_for_mpi();

MPI_Datatype create_particle_for_mpi(MPI_Datatype, MPI_Datatype);

MPI_Datatype create_config_for_mpi();

#endif

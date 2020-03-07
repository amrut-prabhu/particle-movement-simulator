# Particle Movement Simulator

Authors: [Amrut Prabhu](https://github.com/amrut-prabhu) and [Anubhav](https://github.com/anubh-v)

This repository includes 3 parallel computing implementations to simulate the movement of circular particles within a 2D square surface for a certain number of time steps. Particles may change their velocities (speed and direction) upon collising with other particles or colliding with the walls of the square.

## Problem Description and Results

Refer to the following files:

- [`Assignment_Description_1.pdf`](./Assignment_Description_1.pdf) (Sequential, OpenMP, and CUDA programs)
- [`Assignment_Description_2.pdf`](./Assignment_Description_2.pdf) (OpenMPI program)
- [`2dcollisions.pdf`](./2dcollisions.pdf)

Reports are available for each Parallel program (OpenMP and CUDA) and Distributed Message Passing (OpenMPI) implementation:

- [`openmp_report.pdf`](./openmp_report.pdf)
- [`cuda_report.pdf`](./cuda_report.pdf)
- [`mpi_report.pdf`](./mpi_report.pdf)

## Directory Structure

Each of the subdirectories have their own READMEs containing the instructions for compilation and execution. 
They also contain the same input and expected output files in `input/` and `output/`.

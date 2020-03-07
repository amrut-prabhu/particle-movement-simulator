# Particle Simulator (Using MPI)

## Compiling

```sh
mpiCC -std=c++0x resolution.cpp particle.cpp input.cpp utils.cpp simulator.cpp collision.cpp collisions.cpp vector.cpp square.cpp -o main
```

## Running test cases

A minimum of 2 processes is required for execution.
We ran this on mainly Intel Xeon machines and some Intel i7 machines.

The provided `machinefile` contains:
- soctf-pdc-003
- soctf-pdc-004
- soctf-pdc-005

```sh
mpirun -machinefile machinefile -np 60 -use-hwthread-cpus --map-by node ./main < input/input1.in > input/input1.o

# Compare with actual output in output/

diff input/input1.o output/input1.out
```
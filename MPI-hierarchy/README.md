# Particle Movement Simulator (using MPI)

### Compiling

```sh
mpiCC -std=c++0x resolution.cpp particle.cpp input.cpp utils.cpp simulator.cpp collision.cpp collisions.cpp vector.cpp square.cpp -o main
```

### Executing

`(M * (M + 1) / 2) + K + 1` processes is required for execution where `M` is the number of partitions and `K` is the number of nodes used.
The convention for the corresponding rankfile and machinefile are `rankfileK.M` and `machinefileK`.

This approach does not support trivial test cases. 
The number of particles must be at least M.

```sh
mpirun -machinefile machinefile -rankfile rankfile -np <num_processes> ./main < input_file

# Example
mpirun -machinefile machinefile2 -rankfile rankfile2.4 -np 13 ./main < input/input1.in > input/input1.o

# Compare with actual output in output/

diff input/input1.o output/input1.out
```

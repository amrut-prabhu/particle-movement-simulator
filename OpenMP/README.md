# Particle Collisions Simulator (Using OpenMP)

## Running the simulator

### Compilation

**Using Makefile:**
```sh
make
```

**Manual compilation:**
```sh
g++ -std=c++0x -fopenmp resolution.cpp particle.cpp input.cpp utils.cpp simulator.cpp collision.cpp collisions.cpp vector.cpp square.cpp -o main
```

### Execution

To get the execution time of the program in the output, uncomment `#define MEASURE
` (line 10) in `simulator.cpp`.

To test the program with different number of threads, edit the global `NUM_THREADS` variable in `simulator.cpp`.

**Using input/output redirection:**
```sh
./main < input/input1.in > input/input1.o

# Compare with actual output in output/
diff input/input1.o output/input1.out
```

**Execute all input files in directory:**

The command below executes the simulation for all input files (`.in`) in the specified test directory.
A `/out` directory is created in the test directory to store the output files (`.out` by default).

```sh
bash test.sh <test_directory> <optional_output_file_extension>

# Example: Creates `.out` files in input/out 
bash test.sh input out
```

### Test cases

Three tests cases are present in `input/` and their corresponding outputs are stored in `output/`.

- `input1.in` - collisions between 2 particles. At the start, one particle is
  moving vertically upwards while the other moves horizontally leftwards.
- `input2.in` - 4000 particles, medium square
- `input3.in` - 4000 particles, small square (high density)

To get the execution time of the program in the output, uncomment `#define MEASURE
` (line 10) in `simulator.cpp`.

To test the program with different number of threads, edit the global `NUM_THREADS` variable in `simulator.cpp`.

## Data structures

```c
typedef struct {
    int N; // Number of particles on the square surface
    double L; // Size of the square
    double r; // Radius of the particle
    int S; // Number of steps (time units) to run the simulation for
    bool shouldPrint; // whether the position of each particle needs to be printed at each step
    Particle* particles; // array of N particles
} Input;
```

```c
typedef struct {
    int i; // the index of the particle from 0 to N-1
    int particleCollisions; // number of collisions with other particles
    int wallCollisions; // number of collisions with the walls
    Vector position; // current position of particle
    Vector velocity; // current velocity of particle
    Collision* currentCollision; // collision in current step (if any) OR nullptr
    Vector newPosition; // particle's position at start of next step
    Vector newVelocity; // particle's velocity at start of next step
} Particle;
```

```c
typedef struct {
    double x; // x component of the vector
    double y; // y component of the vector
} Vector;
```

```c
typedef struct {
    int A;
    int B; // index of particles that are involved in this collision (A < B)
    double collisionTime; // the time of collision (between 0 and 1)
} Collision;
```

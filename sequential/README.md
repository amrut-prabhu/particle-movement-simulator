# Particle Collisions Simulator

## Running the simulator

### Compilation

**Using Makefile:**
```sh
make
```

**Manual compilation:**
```sh
g++ -std=c++0x resolution.cpp particle.cpp input.cpp utils.cpp simulator.cpp collision.cpp collisions.cpp vector.cpp square.cpp -o main
```

### Execution

**Using input redirection:**
```sh
./main < test/test1.in
```

**Execute all input files in directory:**

The command below executes the simulation for all input files (`.in`) in the specified test directory.
A `/out` directory is created in the test directory to store the output files (`.out` by default).

```sh
./test.sh <test_directory> <optional_output_file_extension>

# Example
# Creates `.output` files in test/particle-density-scaling/out 
./test.sh test/particle-density-scaling output
```

## Resources

[FAQ](https://docs.google.com/document/d/1dJs3E7jvWXBgv1yWNvS2jtgS-LRGlAFsqsp7VRqDlSg/edit)

[Circle-Circle Collision Tutorial](http://ericleong.me/research/circle-circle/)

[Fast, Accurate Collision Detection Between Circles or Spheres](https://www.gamasutra.com/view/feature/131424/pool_hall_lessons_fast_accurate_.php?page=2)

### Test cases

For performance comparisons (execution time) between different implementations and parallelised implementation:
Take 3 measurements, and choose the lowest value.

- `N`: small, medium and large
- `print` vs `perf`
- `L`: small vs large- does it affect number of collisions? (since velocity range is a function of L)
- `r`: small, medium, large
- More wallCollisions vs more particleCollisions

- Parallelise detection per:
    - thread
    
- Parallelise resolution by:
    - particle
    - collision

## Data format

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
## Debugging 

### GDB

Can set breakpoints and run through code line by line.
Good starting instructions [here](https://kb.iu.edu/d/aqsy).

### Memory-related errors

Use [`valgrind`](http://valgrind.org/docs/manual/quick-start.html).

Installation:
```sh
sudo apt install valgrind
```

Usage:

Note that the memory leak detector causes programs to run much slower (eg. 20 to 30 times) than normal.

```sh
# Instead of 
./main < test/input4.in

# Use 
valgrind --leak-check=yes ./main < test/input4.in 
```

The output contains the file names and line numbers that cause memory errors.

### VSCode

Possible changes required for:
- `.vscode/c_cpp_properties.json`
    - `compilerPath`
- `.vscode/tasks.json`
    - `args` (for .cpp files to be included in compilation)
- `.vscode/launch.json`
    - `miDebuggerPath`
    - `args` (for input redirection file)

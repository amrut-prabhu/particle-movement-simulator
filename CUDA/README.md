# Particle Simulator (Using CUDA)

## Compiling

Note: `nvcc` is located at `/usr/local/cuda/bin/nvcc` on SoC Compute Cluster accounts.

`nvcc -std=c++11 -O0 -Xcicc -O0 -Xptxas -O0 -dc main.cu utils.cu particle.cu collisions.cu collision.cu square.cu vector.cu resolution.cu`

`nvcc -o main main.o utils.o particle.o collisions.o collision.o square.o vector.o resolution.o`

## Running test cases

```
./main < input/input1.in > input/input1.o

# Compare with actual output in output/

diff input/input1.o output/input1.out
```
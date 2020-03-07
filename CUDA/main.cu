#include "main.h"

#define THREADS_PER_BLOCK 64

typedef enum {
    MODE_PRINT,
    MODE_PERF
} simulation_mode_t;

__constant__ int n, l, R, s;
int host_n, host_l, host_r, host_s;

__managed__ Particle* particles;

__managed__ Collision* allPossibleCollisions;

__host__ void randomise_particles()
{
    // Generate random positions and velocities for all particles
    srand (time(NULL));

    for (int i = 0; i < host_n; i++)
    {
        particles[i].i = i;
        particles[i].position.x = rand_from(host_r, host_l - host_r);
        particles[i].position.y = rand_from(host_r, host_l - host_r);
        particles[i].velocity.x = rand_from(host_l / 4, host_l / (8 * host_r));
        particles[i].velocity.y = rand_from(host_l / 4, host_l / (8 * host_r));
        particles[i].particleCollisions = 0;
        particles[i].wallCollisions = 0;
    }
}

__host__ void print_particles(int step)
{
    int i;
    for (i = 0; i < host_n; i++) {
        printf("%d %d %10.8f %10.8f %10.8f %10.8f\n", step, i, particles[i].position.x, particles[i].position.y,
            particles[i].velocity.x, particles[i].velocity.y);
    }
}

__host__ void print_statistics(int num_step)
{
    int i;
    for (i = 0; i < host_n; i++) {
        printf("%d %d %10.8f %10.8f %10.8f %10.8f %d %d\n", num_step, i, particles[i].position.x,
            particles[i].position.y, particles[i].velocity.x, particles[i].velocity.y,
            particles[i].particleCollisions, particles[i].wallCollisions);
    }
}

int main(int argc, char** argv)
{

    int i;
    double x, y, vx, vy;
    int num_blocks, num_threads, chunk_size;
    int step;

    simulation_mode_t mode;
    char mode_buf[6];

    chunk_size = 1;

    scanf("%d", &host_n);
    scanf("%d", &host_l);
    scanf("%d", &host_r);
    scanf("%d", &host_s);
    scanf("%5s", mode_buf);

    int maxNumCollisions = (int) ((host_n * (host_n + 1)) / 2.0);
    int total_num_threads = (maxNumCollisions / chunk_size);
    num_blocks = (total_num_threads / THREADS_PER_BLOCK) + 1;  

    cudaMallocManaged(&particles, sizeof(Particle) * host_n);
    cudaMallocManaged(&allPossibleCollisions, sizeof(Collision) * maxNumCollisions);

    for (i = 0; i < host_n; i++) {
        particles[i].i = -1;
        particles[i].particleCollisions = 0;
        particles[i].wallCollisions = 0;
    }

    while (scanf("%d %lf %lf %lf %lf", &i, &x, &y, &vx, &vy) != EOF) {
        particles[i].i = i;
        particles[i].position.x = x;
        particles[i].position.y = y;
        particles[i].velocity.x = vx;
        particles[i].velocity.y = vy;
    }

    if (particles[0].i == -1) {
        randomise_particles();
    }

    mode = strcmp(mode_buf, "print") == 0 ? MODE_PRINT : MODE_PERF;

    /* Copy to GPU constant memory */
    cudaMemcpyToSymbol(n, &host_n, sizeof(n));
    cudaMemcpyToSymbol(l, &host_l, sizeof(l));
    cudaMemcpyToSymbol(R, &host_r, sizeof(R));
    cudaMemcpyToSymbol(s, &host_s, sizeof(s));

    // printf("calling with chunk size %d, num blocks %d, %d total threads\n", chunk_size, num_blocks, total_num_threads);

    for (step = 0; step < host_s; step++) {
        if (mode == MODE_PRINT || step == 0) {
            print_particles(step);
        }
 
        /* Call the kernel */
        detectCollisions<<<num_blocks, THREADS_PER_BLOCK>>>(maxNumCollisions, chunk_size);

        /* Barrier */
        cudaDeviceSynchronize();
 
        cudaError_t rc = cudaGetLastError();
        if (rc != cudaSuccess)
            printf("Last CUDA error %s\n", cudaGetErrorString(rc));

	// We should end up with a set of collisions stored in global/unified memory
	// Retrieve collisions and place into a vector
	std::vector<Collision> collisionsVect;

	for (int i = 0; i < maxNumCollisions; i++)
	{
	   if (allPossibleCollisions[i].B != -99)
	   {
               collisionsVect.push_back(allPossibleCollisions[i]);
	   }
	}
        
	sortAndFilterCollisions(collisionsVect, host_n, particles);

        resolveCollisions(particles, host_n, host_l, host_r);

	// update the particle positions / velocities
        for (int i = 0; i < host_n; i++)
        {
            particles[i].velocity = particles[i].newVelocity;
            particles[i].position = particles[i].newPosition;
            if ((particles[i].position.x < 0) || (particles[i].position.y < 0))
            {
                printf("Invalid position at particle %d\n", i);
            }
        }

    }

    print_statistics(host_s);

    cudaFree(particles);
    cudaFree(allPossibleCollisions);

    return 0;
}

#include <algorithm>
#include <limits>
#include <stdlib.h>
#include <vector>  

#include "collision.h"
#include "collisions.h"
#include "particle.h"
#include "vector.h"
#include "square.h"
#include "utils.h"

using namespace std;

__constant__ extern int n, l, R;
__managed__ extern Particle* particles;
__managed__ extern Collision* allPossibleCollisions;

/**
 * Populate a given vector with all possible collisions in the next time step.
 */
__global__ void detectCollisions(int maxNumCollisions, int chunk_size)
{
    int N = n;
    double L = l;
    double r = R;

    int threadIdInBlock = (threadIdx.x * blockDim.y * blockDim.z) + (threadIdx.y * blockDim.z) + threadIdx.z;
    int blockIdInGrid = (blockIdx.x * gridDim.y * gridDim.z) + (blockIdx.y * gridDim.z) + blockIdx.z;
    int tid = (blockIdInGrid * blockDim.x * blockDim.y * blockDim.z) + threadIdInBlock;

    int startElement = tid*chunk_size;
    int endElement = (maxNumCollisions - (startElement + chunk_size)) >= chunk_size ? startElement + chunk_size : maxNumCollisions;
    for (int k = startElement; k < endElement; k++)
    { 
        int i = k / N;
        int j = k % N;
        
        if (i > j)
        {
            i = N - i;
	    j = N - j - 1;
        }


        if (i != j) 
        {
            Collision* particleCollision = getParticleCollision(particles, i, j, r);
       
            if (particleCollision != nullptr)
   	    {	
                // write particleCollision to unified memory
                allPossibleCollisions[k] = *particleCollision;
                free(particleCollision);
                particleCollision = nullptr;
	    } else
	    {
                allPossibleCollisions[k] = makeNullCollision();
	    }

        } else
        {
        
            Collision* wallCollision = getWallCollision(particles + i, L, r);

            if (wallCollision != nullptr)
            {
                allPossibleCollisions[k] = *wallCollision;
                free(wallCollision);
                wallCollision = nullptr;
	    } else
            {
	        allPossibleCollisions[k] = makeNullCollision();	
	    }
        }
    }
}

void sortAndFilterCollisions(std::vector<Collision> possibleCollisions, int numParticles, Particle* particles)
{
    int N = numParticles;

    // Update the currentCollision of each particle
	sort(possibleCollisions.begin(), possibleCollisions.end(), collisionComparator);

	bool isCurrentCollisionUpdated[N];
	for (int i = 0; i < N; i++)
        {
    	    isCurrentCollisionUpdated[i] = false;
        }

    for (unsigned int i = 0; i < possibleCollisions.size(); i++)
    {
    	int idx1 = possibleCollisions[i].A;
    	int idx2 = possibleCollisions[i].B;

	

    	if (isCurrentCollisionUpdated[idx1] || (idx2 >=0 && isCurrentCollisionUpdated[idx2]))
        {
            continue;
        }

        (particles + idx1)->currentCollision = deepCopy(possibleCollisions[i]);
        isCurrentCollisionUpdated[idx1] = true;
        
        if (idx2 >= 0)
        {
            (particles + idx2)->currentCollision = deepCopy(possibleCollisions[i]);
            isCurrentCollisionUpdated[idx2] = true;
        }
    }
}

/**
 * Returns the info about the collision between these two particles, if any.
 *
 * Return:
 * 	- nulltpr if there is no collision. 
 * 	- Else, Collision with:
 *		- time value between 0 and 1
 *		- distance value
 *		- Indices of particles involved
 */ 
__device__ Collision* getParticleCollision(Particle* particles, int idx1, int idx2, int r)
{
	Particle* p1 = particles + idx1;
	Particle* p2 = particles + idx2;

	Vector* A = &p1->position;
	Vector* B = &p2->position;

	// Velocity of p1 wrt p2, i.e., p2 is assumed to be stationary
	Vector* moveVec = subtract_vectors(&p1->velocity, &p2->velocity);

    // Check 1: Check that A can travel enough to potentially reach B, in this time step
    Vector* C = subtract_vectors(B, A);
    double sumRadii = 2 * r;
    double minDistanceNeeded = norm(C) - sumRadii;

    double moveVecNorm = norm(moveVec);

    if (moveVecNorm < minDistanceNeeded) 
    {
    	free(moveVec);
    	free(C);
        moveVec = nullptr;
        C = nullptr;
    	return nullptr;
    }

    // Check 2: Check that A is moving towards B
    Vector* moveVecNormalised = normalised_vector(moveVec);
    double D = dot_product(C, moveVecNormalised);

    if (D <= 0)	
    {
    	free(moveVec);
    	free(C);
    	free(moveVecNormalised);
        moveVec = nullptr;
        C = nullptr;
        moveVecNormalised = nullptr;
    	return nullptr;
    }

    // Check 3: Check that A gets close enough to B for a possible collision
    double F = norm_squared(C) - D * D;
    double sumRadiiSquared = sumRadii * sumRadii;

    if ((F - sumRadiiSquared) > EPSILON) 
    {
    	free(moveVec);
    	free(C);
    	free(moveVecNormalised);
        moveVec = nullptr;
        C = nullptr;
        moveVecNormalised = nullptr;
        return nullptr;
	}


    // Now, we know that a collision is possible, but it is not guaranteed
    double T = sumRadiiSquared - F;
    if (T < 0) 
    {
    	free(moveVec);
    	free(C);
    	free(moveVecNormalised);
        moveVec = nullptr;
        C = nullptr;
        moveVecNormalised = nullptr;
    	return nullptr;
    }

    double distance = D - sqrt(T);

    if (moveVecNorm < distance) 
    {
    	free(moveVec);
    	free(C);
    	free(moveVecNormalised);
        moveVec = nullptr;
        C = nullptr;
        moveVecNormalised = nullptr;
    	return nullptr;
    }

    // TODO? assumes distance>=0, i.e., the particles have not overlapped

	// Movement vector that will cause the particles to just touch
    Vector* moveVecCollision = scalar_multiple(moveVecNormalised, distance);

    // Fill in and return the collision details
    Collision* collision = (Collision*)(malloc(sizeof(Collision)));
    collision->A = min(p1->i, p2->i);
    collision->B = max(p1->i, p2->i);
    collision->collisionTime = norm(moveVecCollision) / moveVecNorm;

    free(moveVec);
    free(C);
    free(moveVecNormalised);
    free(moveVecCollision);
    moveVec = nullptr;
    C = nullptr;
    moveVecNormalised = nullptr;
    moveVecCollision = nullptr;

    return collision;
}

/**
 * Check whether Particle A collides with any of the 4 walls. If yes,
 * return a pointer to the Collision.
 * If no, return a null pointer.
 */
__device__ Collision* getWallCollision(Particle* A, double squareSize, double radius)
{

    /* determine the point that Particle A could reach, if it travelled
       unimpeded in the current time step */

    Vector* destination = getDestination(A, 1);

    // verify that this destination falls outside the square's bounds
    if (!isOutOfBounds(destination, squareSize, radius))
    {
        free(destination);
        destination = nullptr;
        return nullptr;
    }

    /* find the collision where this particle hits a wall */
    Collision* collision = identifyWallCollision(A, squareSize, radius);

    free(destination);
    destination = nullptr;
    return collision;
}

__host__ __device__ void printCollision(Collision c)
{   
    printf("Before printing!");
    printf("--%d %d %lf\n", c.A, c.B, c.collisionTime);

}

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

/**
 * Populate a given vector with all possible collisions in the next time step.
 */
std::vector<Collision> detectCollisions(Input* input)
{
    int N = input->N;
    double L = input->L;
    double r = input->r;
    Particle* particles = input->particles;

	// Detect all possible collisions
	vector<Collision> possibleCollisions;
    for(int i = 0; i < N; i++) 
    {
        for(int j = i + 1; j < N; j++) 
        {
            Collision* particleCollision = getParticleCollision(particles, i, j, r);
            if (particleCollision == nullptr) {
                continue;
            }

            possibleCollisions.push_back(*particleCollision);
        
            free(particleCollision);
            particleCollision = nullptr;
        }

        Collision* wallCollision = getWallCollision(particles + i, L, r);
        if (wallCollision != nullptr)
        {
            possibleCollisions.push_back(*wallCollision);

            free(wallCollision);
            wallCollision = nullptr;
        }
    }

    return possibleCollisions;
		
}

void sortAndFilterCollisions(std::vector<Collision> possibleCollisions, Input* input)
{
    int N = input->N;
    Particle* particles = input->particles;

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
Collision* getParticleCollision(Particle* particles, int idx1, int idx2, int r)
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
    collision->A = idx1;
    collision->B = idx2;
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
Collision* getWallCollision(Particle* A, double squareSize, double radius)
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

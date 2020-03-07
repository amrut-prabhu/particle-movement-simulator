#include "square.h"

/**
 * Return true iff a given position is outside the square.
 */
__device__ __host__ bool isOutOfBounds(Vector* position, double squareSize, double radius)
{
    return hasExceededTopWall(position, squareSize, radius) || 
            hasExceededBottomWall(position, radius) ||
            hasExceededRightWall(position, squareSize, radius) ||
            hasExceededLeftWall(position, radius);
}

__device__ __host__ bool hasExceededTopWall(Vector* position, double squareSize, double radius) 
{
    return ((position->y) + radius) > squareSize;
}

__device__ __host__ bool hasExceededBottomWall(Vector* position, double radius) {
    return ((position->y) - radius) < 0;
}

__device__ __host__ bool hasExceededRightWall(Vector* position, double squareSize, double radius) {
    return ((position->x) + radius) > squareSize;
}

__device__ __host__ bool hasExceededLeftWall(Vector* position, double radius) {
    return ((position->x) - radius) < 0;
}


/**
 * Given a particle, identify the particle's collision point with a wall. The collision point
 * is the point where the particle is just touching the wall, right before the
 * collision.
 *
 * Precondition: The particle will definitely collide with one of the walls.
 */
__device__ __host__ Collision* identifyWallCollision(Particle* A, double squareSize, double radius)
{
    Pair collisionInfo = identifyWallCollision(&(A->position), &(A->velocity),
                                                                     squareSize, radius);
    
    // allocate and return Collision
    Collision* wallCollision = (Collision*)malloc(sizeof(Collision));
    wallCollision->A = A->i;
    wallCollision->B = collisionInfo.collisionType;
    wallCollision->collisionTime = collisionInfo.collisionTime;
    return wallCollision;
}


__device__ __host__ Pair identifyWallCollision(Vector* position, Vector* velocity, double squareSize,
                                                  double radius) {

    double timeToCollideVertically = getTimeForVerticalWallCollision(position, velocity,
                                                                         squareSize, radius);

    double timeToCollideHorizontally = getTimeForHorizontalWallCollision(position, velocity,
                                                                            squareSize, radius);
    
    int wallIndex;
    /* wallIndex = -1 --> collision with top or bottom wall
       wallIndex = -2 --> collision with right or left wall
       wallIndex = -3 --> collision with one of the corners
    */

   if (isEqual(timeToCollideHorizontally, timeToCollideVertically))
   {
       wallIndex = -3;
   } else if (timeToCollideHorizontally > timeToCollideVertically)
   {
       wallIndex = -2;
   } else {
       wallIndex = -1;
   }

   double collisionTime = min(timeToCollideVertically, timeToCollideHorizontally);
   return Pair {collisionTime, wallIndex};                       
}

/**
 * Return the time needed for the particle to collide with the vertical walls.
 * i.e. either the left wall or the right wall
 */
__device__ __host__ double getTimeForVerticalWallCollision(Vector* position, Vector* velocity, 
                                           double squareSize, double radius)
{
    return getTimeForOneDimWallCollision(position->x, velocity->x, squareSize, radius);
}

/**
 * Return the time needed for the particle to collide with the horizontal walls.
 * i.e. either the top wall or the bottom wall.
 */
__device__ __host__ double getTimeForHorizontalWallCollision(Vector* position, Vector* velocity, 
                                            double squareSize, double radius)
{
    return getTimeForOneDimWallCollision(position->y, velocity->y, squareSize, radius);
}

/** 
 * Return the amount of time needed for the particle to collide with the walls
 * in the chosen dimension.
 */
__device__ __host__ double getTimeForOneDimWallCollision(double oneDimPosition, double oneDimVelocity, 
                                         double squareSize, double radius) {

    double time;

    if (isEqual(oneDimVelocity, 0))
    {
        // particle is stationary in the chosen dimension
        return 100000000;
    } else if (oneDimVelocity > 0)
    {
        // particle moving towards positive direction in chosen dimension
        // either right wall or top wall
        double distanceToPositiveWall = squareSize - (oneDimPosition + radius);
        time = (distanceToPositiveWall / oneDimVelocity);
    } else {
        // xVelocity < 0
        // particle moving towards negative direction in chosen dimension
        // either left wall or bottom wall
        double distanceToNegativeWall = oneDimPosition - radius;
        time = (distanceToNegativeWall / oneDimVelocity);
    }

    return time < 0 ? -1 * time : time;
}

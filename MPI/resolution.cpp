#include "resolution.h"
#include "utils.h"

void resolveCollisions(Config* config, Particle* particles)
{
    int numParticles = config->N;
    double squareSize = config->L;
    double radius = config->r;

    // Iterate through all particles
    for (int i = 0; i < numParticles; i++)
    {
        Particle* currentParticle = particles + i;
        if (hasCollision(currentParticle))
        {
            // retrieve the Collision
            Collision* collision = currentParticle->currentCollision;

            if (hasCollisionWithParticle(currentParticle))
            {
                // particle collided with another particle

                // retrieve the other particle involved in this collision
                Particle* collidingPartner = getCollidingPartner(particles,
                                                          collision, currentParticle);

                // resolve this particle-particle collision

                // find the particle's velocity after the collision
                resolveVelocity(currentParticle, collidingPartner);
                // find the particle's position at the end of this time step
                resolvePosition(currentParticle, collision, squareSize, radius);

                currentParticle->particleCollisions++;
            } else
            {
                // particle collided with a wall

                /* The particle's new velocity will simply be the
                   reverse of its velocity before the collision */
                resolveVelocity(currentParticle, collision);

                // find the particle's position at the end of this time step
                resolvePosition(currentParticle, collision, squareSize, radius);

                currentParticle->wallCollisions++;
            }

            currentParticle->currentCollision = nullptr;
            free(collision);
            collision = nullptr;

        } else
        {
            /* If particle is not involved in a collision, its velocity
             * will remain unchanged.
             */
            currentParticle->newVelocity = currentParticle->velocity;

            /* Find the point that this particle will reach at the end of the
             * current time step.
             */
            Vector* destination = getDestination(currentParticle, 1);

            // update particle's position
            (currentParticle->newPosition).x = destination->x;
            (currentParticle->newPosition).y = destination->y;

            free(destination);
            destination = nullptr;
        }

    }
}


/*
 * Given an array of particles, a collision and one of the particles in this
 * collision, return a pointer to the OTHER particle in this collision.
 *
 * Precondition: The given collision must be a particle-particle collision.
 */
Particle* getCollidingPartner(Particle* particles, Collision* collision,
                                  Particle* particleA)
{
    if ((particleA->i) == (collision->A))
    {
        return particles + (collision->B);
    } 
    else
    {
        return particles + (collision->A);
    }
}

/*
 * Given a particle A and its colliding partner, find particle A's
 * velocity after the collision and update particle A.
 */
void resolveVelocity(Particle* toBeResolved, Particle* partner)
{
    // get the vector that's normal to surface of collision
    Vector* normal = getConnectingVector(toBeResolved, partner);

    // normalise the normal vector
    Vector* unit_normal = normalised_vector(normal);

    // get the vector that's tangent to surface of collision
    Vector* unit_tangent = perpendicular_vector(unit_normal);

    Vector* old_velocity = &(toBeResolved->velocity);
    Vector* partner_old_velocity = &(partner->velocity);

    // project velocity along unit tangent
    Vector* tangent_velocity = project_along_unit(old_velocity, unit_tangent);

    // project velocity along unit normal
    Vector* normal_velocity = project_along_unit(old_velocity, unit_normal);

    Vector* partner_normal_velocity = project_along_unit(partner_old_velocity,
                                                             unit_normal);
    // tangent velocity remains unchanged
    // modify the normal velocity according to physics
    free(normal_velocity);
    normal_velocity = partner_normal_velocity;

    // add tangent and normal velocities to get overall velocity
    Vector* new_velocity = add(normal_velocity, tangent_velocity);

    // Update the particle
    toBeResolved->newVelocity = *new_velocity;

    // free the vectors created during this function application
    free(normal);
    free(unit_normal);
    free(unit_tangent);
    free(tangent_velocity);
    free(partner_normal_velocity);
    free(new_velocity);
    normal = nullptr;
    unit_normal = nullptr;
    unit_tangent = nullptr;
    tangent_velocity = nullptr;
    partner_normal_velocity = nullptr;
    new_velocity = nullptr;
}

/** Given a particle A and its Collision, find A's position at the end
 *  of the current time interval, and update A.
 *
 *  Precondition: Particle A should already know its velocity after the
 *  collision.
 */
void resolvePosition(Particle* toBeResolved, Collision* collision, double squareSize,
                         double radius)
{

    // retrieve the time of collision
    double collisionTime = collision->collisionTime;
    double timeRemaining = 1 - collisionTime;

    /* use collision time to find out this particle's position
       at the moment of collision */
    Vector* collisionPoint = getDestination(toBeResolved, collisionTime);

    // get particle's velocity after collision
    Vector* newVelocity = &(toBeResolved->newVelocity);

    /* calculate the distance that the particle will travel parallel after
     * the collision */

    double xTravelDistance = timeRemaining*(newVelocity->x);
    double yTravelDistance = timeRemaining*(newVelocity->y);

    // Check whether the particle can technically collide with another
    // wall after its first collision.

    Vector* newPosition = make_vector(collisionPoint->x + xTravelDistance,
                                         collisionPoint->y + yTravelDistance);

    // ensure newPosition is witin the walls
    // if not, adjust it such that the particle is just about to collide with a wall
    if (isOutOfBounds(newPosition, squareSize, radius))
    {
        free(newPosition);
        newPosition = nullptr;

        std::pair<double, int> collisionInfo = identifyWallCollision(collisionPoint, newVelocity,
                                                                         squareSize, radius);
        
        xTravelDistance = collisionInfo.first*(newVelocity->x);
        yTravelDistance = collisionInfo.first*(newVelocity->y);
        newPosition = make_vector(collisionPoint->x + xTravelDistance,
                                      collisionPoint->y + yTravelDistance);
    }

    // update particle
    toBeResolved->newPosition = *newPosition;
   
    free(newPosition);
    free(collisionPoint);
    newPosition = nullptr;
    collisionPoint = nullptr;
}


/**
 * Given a particle, set its new velocity as the opposite of its current
 * velocity. This is useful for resolving collisions with walls.
 */
void resolveVelocity(Particle *toBeResolved, Collision* wallCollision)
{
    // check whether the particle collided with a vertical or horizontal wall
    int wallType = wallCollision->B;
    Vector* currentVelocity = &(toBeResolved->velocity);
    Vector* resolvedVelocity;

    if (wallType == -1)
    {
        // particle collided with a horizontal wall
        // simply reverse the y component of the velocity
        resolvedVelocity = make_vector(currentVelocity->x, -1*(currentVelocity->y));
    } 
    else if (wallType == -2)
    {
        // particle collided with a verticall wall
        // simply reverse the x component of the velocity
        resolvedVelocity = make_vector(-1*(currentVelocity->x), currentVelocity->y);
    } 
    else if (wallType == -3)
    {
        // particle collided with a corner
        // reverse both components of velocity
        resolvedVelocity = make_vector(-1*(currentVelocity->x), -1*(currentVelocity->y));
    }

    // update particle
    toBeResolved->newVelocity = *resolvedVelocity;
    free(resolvedVelocity);
    resolvedVelocity = nullptr;
}

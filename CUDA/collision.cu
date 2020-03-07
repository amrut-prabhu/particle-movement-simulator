#include <stdlib.h>
#include <math.h>

#include "collision.h"
#include "utils.h"

Collision* deepCopy(Collision c)
{
    Collision* copy = (Collision*)(malloc(sizeof(Collision)));

    copy->A = c.A;
    copy->B = c.B;
    copy->collisionTime = c.collisionTime;

    return copy;
}

/**
 * Returns true if c1 should occur before c2. False otherwise.
 */
bool collisionComparator(Collision c1, Collision c2)
{
	if (isEqual(c1.collisionTime , c2.collisionTime))
	{
		if (c1.A == c2.A)
		{
			return c1.B < c2.B;
		}

		return c1.A < c2.A;
	}

	return c1.collisionTime < c2.collisionTime;
}

__host__ __device__ Collision makeNullCollision()
{
    Collision nullCollision; 
    nullCollision.B = -99;
    return nullCollision;
}

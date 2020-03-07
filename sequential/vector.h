#ifndef VECTOR_H
#define VECTOR_H

#include <math.h>
#include <cstdlib>

typedef struct {
    double x; // x component of the vector
    double y; // y component of the vector
} Vector;

#endif

/**
 * Returns the norm (magnitude) of the input vector.
 */
double norm(Vector*);

/**
 * Returns the square of the norm (magnitude) of the input vector.
 */
double norm_squared(Vector*);

/**
 * Returns the unit vector obtaining by normalising the input vector.
 */
Vector* normalised_vector(Vector*);

/**
 * Returns the difference between the input vectors (v1 - v2).
 */
Vector* subtract_vectors(Vector*, Vector*);

/**
 * Returns the scalar dot product of the two input vectors.
 */
double dot_product(Vector*, Vector*);

/**
 * Return a vector that is obtained by multiplying a given vector by a scalar.
 */
Vector* scalar_multiple(Vector*, double);

/**
 * Negate an existing vector.
 */
Vector* negate(Vector*);

/**
 * Return the sum of 2 vectors.
 */
Vector* add(Vector*, Vector*);

/**
 * Returns a vector that's perpendicular to the input vector.
 */
Vector* perpendicular_vector(Vector*);

/**
 * Return the projection of a given vector along the direction of a
 * given UNIT vector.
 */
Vector* project_along_unit(Vector*, Vector*);

Vector* translate(Vector*, Vector*, double);

Vector* make_vector(double, double);

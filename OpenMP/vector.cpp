#include "vector.h"
#include "utils.h"

/**
 * Returns the square of the norm (magnitude) of the input vector.
 */
double norm_squared(Vector* v)
{
	return (v->x * v->x) + (v->y * v->y);
}

/**
 * Returns the norm (magnitude) of the input vector. 
 * Note that the sqrt operation is computationally expensive.
 */
double norm(Vector* v)
{
	return sqrt(v->x * v->x + v->y * v->y);
}

/**
 * Returns the unit vector obtaining by normalising the input vector.
 */
Vector* normalised_vector(Vector* v)
{
  double magnitude = norm(v);

  Vector* v_n = (Vector*)(malloc(sizeof(Vector)));

  if (isEqual(magnitude, 0)) {
    v_n->x = v->x;
    v_n->y = v->y;
  } else {
    v_n->x = v->x / magnitude;
    v_n->y = v->y / magnitude;
  }

  return v_n;
}

/**
 * Returns the difference between the input vectors (v1 - v2).
 */
Vector* subtract_vectors(Vector* v1, Vector* v2)
{
	Vector* v = (Vector*)(malloc(sizeof(Vector)));

	v->x = v1->x - v2->x;
	v->y = v1->y - v2->y;

	return v;
}

/**
 * Returns the scalar dot product of the two input vectors.
 */
double dot_product(Vector* v1, Vector* v2)
{
	return (v1->x * v2->x) + (v1->y * v2->y);
}

/**
 * Return a vector obtained by multiplying a given vector by a scalar.
 */
Vector* scalar_multiple(Vector* v, double scalar) {
  Vector* multiple = (Vector*)(malloc(sizeof(Vector)));
  multiple->x = (v->x)*scalar;
  multiple->y = (v->y)*scalar;

  return multiple;
}

/**
 * Negate an existing vector.
 */
Vector* negate(Vector* v)
{
  // TODO: Use bitwise operations here?
  return scalar_multiple(v, -1);
}

/**
 * Return result of adding 2 vectors.
 */
Vector* add(Vector* v1, Vector* v2)
{
  Vector* sum = (Vector*)(malloc(sizeof(Vector)));
  sum->x = v1->x + v2->x;
  sum->y = v1->y + v2->y;

  return sum;
}

/**
 * Returns a vector perpendicular to the input vector.
 */
Vector* perpendicular_vector(Vector* v)
{
  Vector* perpendicular = (Vector*)(malloc(sizeof(Vector)));
  perpendicular->x = (v->y)*(-1);
  perpendicular->y = (v->x)*(-1);

  return perpendicular;
}

/**
 * Return the projection of a given vector along the direction of a
 * given UNIT vector.
 */
Vector* project_along_unit(Vector* v, Vector* unit_v) {

  double projection_magnitude = dot_product(v, unit_v);
  return scalar_multiple(unit_v, projection_magnitude);

}

/**
 * Given vectors OX and OY, return the vector ZY such that
 * |XY| = |XZ| + |ZY|, and |XZ| equals a given scalar d.
 *
 * In other words, translate the point X by d units
 * in the direction of XY.
 */
Vector* translate(Vector* OX, Vector* OY, double shiftLength)
{
    // XY = OY - OX
    Vector* XY = subtract_vectors(OY, OX);
    Vector* normalisedXY = normalised_vector(XY);
    Vector* scaled = scalar_multiple(normalisedXY, shiftLength);
    Vector* ZY = subtract_vectors(XY, scaled);

    free(XY);
    free(normalisedXY);
    free(scaled);

    return ZY;
}

Vector* make_vector(double x, double y) {
    Vector* vector = (Vector*)malloc(sizeof(Vector));
    vector->x = x;
    vector->y = y;
    return vector;
}


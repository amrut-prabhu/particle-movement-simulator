#ifndef SQUARE_H
#define SQUARE_H

#include "particle.h"
#include "utils.h" 
#include <limits>
#include <utility>
#include <algorithm>

bool isOutOfBounds(Vector*, double, double);

bool hasExceededTopWall(Vector*, double, double);

bool hasExceededBottomWall(Vector*, double);

bool hasExceededRightWall(Vector*, double, double);

bool hasExceededLeftWall(Vector*, double);

Collision* identifyWallCollision(Particle*, double, double);

std::pair<double, int> identifyWallCollision(Vector* position, Vector* velocity, double squareSize,
                                                  double radius);

double getTimeForHorizontalWallCollision(Vector*, Vector*, double, double);

double getTimeForVerticalWallCollision(Vector*, Vector*, double, double);

double getTimeForOneDimWallCollision(double, double, double, double);

#endif

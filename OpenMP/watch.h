#ifndef TIMESTATS_H
#define TIMESTATS_H

typedef struct {
    double wall0;
    double cpu0;
    double setup_time;
    double collision_detection_time;
    double collision_sorting_time;
    double collision_resolution_time;
    double particle_update_time; // time to update particles at the end of each step
} Watch;

#endif

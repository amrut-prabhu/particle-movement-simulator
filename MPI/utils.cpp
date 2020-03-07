#include "utils.h"

// Taken from https://stackoverflow.com/a/17440673/8802161
//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time(){
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;
    }
}

//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif

/**
 * Generates a random double in the range of the input- [a, b] or [b, a].
 */
double rand_from(double a, double b) 
{
    double upper = b > a ? b : a;
    double lower = b > a ? a : b;

    double range = (upper - lower); 
    double div = RAND_MAX / range;
    return lower + (rand() / div);
}

/**
 * Test whether two double precision floating points are close enough
 * to be equal
 */
bool isEqual(double a, double b)
{
    // acknowledgement: https://isocpp.org/wiki/faq/newbie#floating-point-arith
    double epsilon = 1e-5;
    return std::abs(a - b) <= epsilon * std::abs(a);

}

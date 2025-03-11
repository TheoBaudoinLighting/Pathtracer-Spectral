#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class Ray {
public:
    Point3 origin;
    Vec3 direction;
    float wavelength;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point3& origin_, const Vec3& direction_, float wavelength_ = 550.0f)
        : origin(origin_), direction(direction_), wavelength(wavelength_) {}

    __host__ __device__ Point3 at(float t) const {
        return origin + t*direction;
    }
};

#endif // RAY_CUH
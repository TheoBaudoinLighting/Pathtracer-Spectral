#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "ray.cuh"
#include <curand_kernel.h>

class Camera {
public:
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius;

    __host__ Camera(
        Point3 lookfrom,
        Point3 lookat,
        Vec3 vup,
        float vfov,
        float aspect_ratio,
        float aperture,
        float focus_dist
    ) {
        float theta = vfov * M_PI / 180.0f;
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - focus_dist * w;
        lens_radius = aperture / 2.0f;
    }

    __device__ Ray get_ray(float s, float t, curandState* local_rand_state, float wavelength) const {
        Vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        Vec3 offset = u * rd.x + v * rd.y;

        return Ray(
            origin + offset,
            lower_left_corner + s*horizontal + t*vertical - origin - offset,
            wavelength
        );
    }
};

#endif // CAMERA_CUH
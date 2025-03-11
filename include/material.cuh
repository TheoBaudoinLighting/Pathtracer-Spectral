#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "ray.cuh"
#include "spectrum.cuh"
#include <curand_kernel.h>

struct HitRecord;

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMISSIVE,
    SPECTRAL
};

struct MaterialData {
    MaterialType type;

    Color albedo;
    float fuzz;
    float ior;
    bool dispersive;
    Color emission;
    int spectral_function_id;
};

__device__ inline float schlick_reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

__device__ inline bool scatter_lambertian(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

    if (scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
    }

    scattered = Ray(rec.p, scatter_direction, r_in.wavelength);
    attenuation = mat_data.albedo;
    return true;
}

__device__ inline bool scatter_metal(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 reflected = reflect(normalize(r_in.direction), rec.normal);
    scattered = Ray(
        rec.p,
        reflected + mat_data.fuzz * random_in_unit_sphere(rand_state),
        r_in.wavelength
    );
    attenuation = mat_data.albedo;
    return (dot(scattered.direction, rec.normal) > 0);
}

__device__ inline bool scatter_dielectric(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    attenuation = Color(1.0f, 1.0f, 1.0f);

    float refraction_ratio;
    float index;

    if (mat_data.dispersive) {
        index = flint_glass_ior(r_in.wavelength);
    } else {
        index = mat_data.ior;
    }

    if (rec.front_face) {
        refraction_ratio = 1.0f / index;
    } else {
        refraction_ratio = index;
    }

    Vec3 unit_direction = normalize(r_in.direction);
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    Vec3 direction;

    if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > curand_uniform(rand_state)) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    }

    scattered = Ray(rec.p, direction, r_in.wavelength);
    return true;
}

__device__ inline bool scatter_emissive(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    return false;
}

__device__ inline bool scatter_spectral(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

    if (scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
    }

    scattered = Ray(rec.p, scatter_direction, r_in.wavelength);

    switch (mat_data.spectral_function_id) {
        case 0:
            if (r_in.wavelength >= 490 && r_in.wavelength <= 570) {
                attenuation = Color(0.8f, 0.8f, 0.8f);
            } else {
                attenuation = Color(0.1f, 0.1f, 0.1f);
            }
            break;
        case 1:
            if (r_in.wavelength >= 600) {
                attenuation = Color(0.9f, 0.9f, 0.9f);
            } else {
                attenuation = Color(0.1f, 0.1f, 0.1f);
            }
            break;
        default:
            attenuation = mat_data.albedo;
            break;
    }

    return true;
}

__device__ inline bool scatter(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    switch (mat_data.type) {
        case LAMBERTIAN:
            return scatter_lambertian(r_in, rec, mat_data, attenuation, scattered, rand_state);
        case METAL:
            return scatter_metal(r_in, rec, mat_data, attenuation, scattered, rand_state);
        case DIELECTRIC:
            return scatter_dielectric(r_in, rec, mat_data, attenuation, scattered, rand_state);
        case EMISSIVE:
            return scatter_emissive(r_in, rec, mat_data, attenuation, scattered, rand_state);
        case SPECTRAL:
            return scatter_spectral(r_in, rec, mat_data, attenuation, scattered, rand_state);
        default:
            return false;
    }
}

__device__ inline Color emit(const MaterialData& mat_data, float u, float v, const Point3& p) {
    if (mat_data.type == EMISSIVE) {
        return mat_data.emission;
    }
    return Color(0.0f, 0.0f, 0.0f);
}

#endif // MATERIAL_CUH
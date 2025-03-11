#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "material.cuh"
#include "camera.cuh"
#include "spectrum.cuh"
#include "cuda_utils.cuh"

__global__ void init_rand_state(curandState* rand_state, int width, int height, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__device__ Color ray_color(
    const Ray& r,
    const HittableData* objects,
    int num_objects,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz,
    const MaterialData* materials,
    int depth,
    curandState* local_rand_state
) {
    Ray cur_ray = r;
    Color cur_attenuation(1.0f, 1.0f, 1.0f);
    Color cur_emitted(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < depth; i++) {
        HitRecord rec;

        if (hit_objects(
                cur_ray, 0.001f, FLT_MAX, rec,
                objects, num_objects,
                spheres, triangles, rect_xy, rect_xz, rect_yz
            )) {

            Color emitted = emit(materials[rec.material_id], rec.u, rec.v, rec.p);
            cur_emitted += cur_attenuation * emitted;

            Ray scattered;
            Color attenuation;

            if (scatter(cur_ray, rec, materials[rec.material_id], attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return cur_emitted;
            }
        } else {
            Vec3 unit_direction = normalize(cur_ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            Color background = (1.0f - t) * Color(0.05f, 0.05f, 0.05f) + t * Color(0.0f, 0.0f, 0.1f);
            return cur_emitted + cur_attenuation * background;
        }

        if (cur_attenuation.length_squared() < 0.001f) {
            float q = fmaxf(fmaxf(cur_attenuation.x, cur_attenuation.y), cur_attenuation.z);
            if (curand_uniform(local_rand_state) >= q) {
                return cur_emitted;
            }
            cur_attenuation /= q;
        }
    }

    return cur_emitted;
}

__global__ void render_kernel(
    Color* framebuffer,
    int width,
    int height,
    int samples_per_pixel,
    int max_depth,
    int spectral_samples,
    float lambda_min,
    float lambda_max,
    Camera* camera,
    HittableData* objects,
    int num_objects,
    SphereData* spheres,
    TriangleData* triangles,
    RectangleXYData* rect_xy,
    RectangleXZData* rect_xz,
    RectangleYZData* rect_yz,
    MaterialData* materials,
    curandState* rand_state
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curandState local_rand_state = rand_state[pixel_index];

    Color pixel_color(0, 0, 0);

    float delta_lambda = (lambda_max - lambda_min) / spectral_samples;

    for (int s = 0; s < samples_per_pixel; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(width - 1);
        float v = float(j + curand_uniform(&local_rand_state)) / float(height - 1);

        Color spectral_color(0, 0, 0);
        for (int w = 0; w < spectral_samples; w++) {
            float lambda = lambda_min + (w + curand_uniform(&local_rand_state)) * delta_lambda;

            Ray r = camera->get_ray(u, v, &local_rand_state, lambda);

            Color ray_contribution = ray_color(
                r, objects, num_objects,
                spheres, triangles, rect_xy, rect_xz, rect_yz,
                materials, max_depth, &local_rand_state
            );

            Color lambda_rgb = wavelength_to_rgb(lambda);
            spectral_color += ray_contribution * lambda_rgb;
        }

        pixel_color += spectral_color / float(spectral_samples);
    }

    rand_state[pixel_index] = local_rand_state;

    framebuffer[pixel_index] = pixel_color;
}

extern "C" void launch_init_rand_states(
    curandState* rand_state,
    const ImageProperties& img_props,
    unsigned long seed
) {
    dim3 blocks(16, 16);
    dim3 grid(
        (img_props.width + blocks.x - 1) / blocks.x,
        (img_props.height + blocks.y - 1) / blocks.y
    );

    init_rand_state<<<grid, blocks>>>(
        rand_state,
        img_props.width,
        img_props.height,
        seed
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

extern "C" void launch_render_kernel(
    Color* framebuffer,
    const ImageProperties& img_props,
    Camera* dev_camera,
    HittableData* dev_objects,
    int num_objects,
    SphereData* dev_spheres,
    TriangleData* dev_triangles,
    RectangleXYData* dev_rect_xy,
    RectangleXZData* dev_rect_xz,
    RectangleYZData* dev_rect_yz,
    MaterialData* dev_materials,
    curandState* rand_state
) {
    dim3 blocks(16, 16);
    dim3 grid(
        (img_props.width + blocks.x - 1) / blocks.x,
        (img_props.height + blocks.y - 1) / blocks.y
    );

    render_kernel<<<grid, blocks>>>(
        framebuffer,
        img_props.width,
        img_props.height,
        img_props.samples_per_pixel,
        img_props.max_depth,
        img_props.spectral_samples,
        img_props.lambda_min,
        img_props.lambda_max,
        dev_camera,
        dev_objects,
        num_objects,
        dev_spheres,
        dev_triangles,
        dev_rect_xy,
        dev_rect_xz,
        dev_rect_yz,
        dev_materials,
        rand_state
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
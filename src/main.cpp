#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "spectrum.cuh"
#include "cuda_utils.cuh"
#include "scene_setup.h"
#include "image_io.h"

extern "C" void launch_init_rand_states(
    curandState* rand_state,
    const ImageProperties& img_props,
    unsigned long seed
);

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
);

int main() {
    ImageProperties img_props;
    img_props.width = 800;
    img_props.height = 800;
    img_props.num_pixels = img_props.width * img_props.height;
    img_props.samples_per_pixel = 512;
    img_props.max_depth = 15;
    img_props.spectral_samples = 25;
    img_props.lambda_min = LAMBDA_MIN;
    img_props.lambda_max = LAMBDA_MAX;

    std::cout << "Render configuration:" << std::endl;
    std::cout << "  Resolution: " << img_props.width << "x" << img_props.height << std::endl;
    std::cout << "  Samples per pixel: " << img_props.samples_per_pixel << std::endl;
    std::cout << "  Maximum depth: " << img_props.max_depth << std::endl;
    std::cout << "  Spectral samples: " << img_props.spectral_samples << std::endl;
    std::cout << "  Spectral range: " << img_props.lambda_min << "-" << img_props.lambda_max << " nm" << std::endl;

    int scene_choice = 0;
    std::cout << "\nChoose a scene:" << std::endl;
    std::cout << "  1. Prism in Cornell Box" << std::endl;
    std::cout << "  2. Prism showcase (exterior)" << std::endl;
    std::cout << "  3. Material showcase" << std::endl;
    std::cout << "Your choice (1-3): ";
    std::cin >> scene_choice;

    Scene scene(Camera(Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 40, 1.0, 0.0, 10.0), img_props);

    switch (scene_choice) {
        case 1:
            scene = create_cornell_box_scene(img_props);
            break;
        case 2:
            scene = create_prism_showcase_scene(img_props);
            break;
        case 3:
            scene = create_material_showcase_scene(img_props);
            break;
        default:
            scene = create_cornell_box_scene(img_props);
            break;
    }

    std::cout << "\nScene information:" << std::endl;
    std::cout << "  Number of objects: " << scene.objects.size() << std::endl;
    std::cout << "  Number of materials: " << scene.materials.size() << std::endl;

    HittableData* dev_objects = nullptr;
    SphereData* dev_spheres = nullptr;
    TriangleData* dev_triangles = nullptr;
    RectangleXYData* dev_rect_xy = nullptr;
    RectangleXZData* dev_rect_xz = nullptr;
    RectangleYZData* dev_rect_yz = nullptr;
    MaterialData* dev_materials = nullptr;
    Camera* dev_camera = nullptr;

    Color* dev_framebuffer = cuda_alloc<Color>(img_props.num_pixels);
    curandState* dev_rand_state = cuda_alloc<curandState>(img_props.num_pixels);

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\nInitializing random number generators..." << std::endl;
    launch_init_rand_states(dev_rand_state, img_props, time(nullptr));

    std::cout << "Allocating and transferring scene data to GPU..." << std::endl;
    allocate_scene_on_gpu(
        scene,
        &dev_objects,
        &dev_spheres,
        &dev_triangles,
        &dev_rect_xy,
        &dev_rect_xz,
        &dev_rect_yz,
        &dev_materials,
        &dev_camera
    );

    std::cout << "Starting render..." << std::endl;
    launch_render_kernel(
        dev_framebuffer,
        img_props,
        dev_camera,
        dev_objects,
        static_cast<int>(scene.objects.size()),
        dev_spheres,
        dev_triangles,
        dev_rect_xy,
        dev_rect_xz,
        dev_rect_yz,
        dev_materials,
        dev_rand_state
    );

    std::vector<Color> framebuffer(img_props.num_pixels);
    cuda_copy_to_host(dev_framebuffer, framebuffer.data(), img_props.num_pixels);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Render completed in " << elapsed.count() << " seconds." << std::endl;

    cuda_free(dev_framebuffer);
    cuda_free(dev_rand_state);
    free_scene_on_gpu(
        dev_objects,
        dev_spheres,
        dev_triangles,
        dev_rect_xy,
        dev_rect_xz,
        dev_rect_yz,
        dev_materials,
        dev_camera
    );

    std::string filename = generate_timestamp_filename("render", "ppm");
    save_image_ppm(filename, framebuffer, img_props.width, img_props.height, img_props.samples_per_pixel);
    
    return 0;
}
#ifndef SCENE_SETUP_H
#define SCENE_SETUP_H

#include <vector>
#include <memory>
#include "hittable.cuh"
#include "material.cuh"
#include "camera.cuh"
#include "cuda_utils.cuh"

struct Scene {
    std::vector<HittableData> objects;
    std::vector<SphereData> spheres;
    std::vector<TriangleData> triangles;
    std::vector<RectangleXYData> rect_xy;
    std::vector<RectangleXZData> rect_xz;
    std::vector<RectangleYZData> rect_yz;

    std::vector<MaterialData> materials;

    Camera camera;

    ImageProperties image_properties;

    Scene(const Camera& cam, const ImageProperties& img_props)
        : camera(cam), image_properties(img_props) {}
};

Scene create_prism_showcase_scene(const ImageProperties& img_props);
Scene create_material_showcase_scene(const ImageProperties& img_props);
Scene create_cornell_box_scene(const ImageProperties& img_props);

void allocate_scene_on_gpu(
    const Scene& scene,
    HittableData** dev_objects,
    SphereData** dev_spheres,
    TriangleData** dev_triangles,
    RectangleXYData** dev_rect_xy,
    RectangleXZData** dev_rect_xz,
    RectangleYZData** dev_rect_yz,
    MaterialData** dev_materials,
    Camera** dev_camera
);

void free_scene_on_gpu(
    HittableData* dev_objects,
    SphereData* dev_spheres,
    TriangleData* dev_triangles,
    RectangleXYData* dev_rect_xy,
    RectangleXZData* dev_rect_xz,
    RectangleYZData* dev_rect_yz,
    MaterialData* dev_materials,
    Camera* dev_camera
);

#endif // SCENE_SETUP_H
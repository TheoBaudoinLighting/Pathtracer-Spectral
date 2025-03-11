#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <string>
#include <vector>
#include "vec3.cuh"

void save_image_ppm(
    const std::string& filename,
    const std::vector<Color>& framebuffer,
    int width,
    int height,
    int samples_per_pixel
);

void save_image_png(
    const std::string& filename,
    const std::vector<Color>& framebuffer,
    int width,
    int height,
    int samples_per_pixel
);

std::string generate_timestamp_filename(const std::string& prefix, const std::string& extension);

#endif // IMAGE_IO_H
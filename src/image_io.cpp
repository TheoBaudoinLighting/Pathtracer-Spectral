#include "image_io.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

void save_image_ppm(
    const std::string& filename,
    const std::vector<Color>& framebuffer,
    int width,
    int height,
    int samples_per_pixel
) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    outfile << "P3\n" << width << " " << height << "\n255\n";
    
    float scale = 1.0f / samples_per_pixel;
    
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int pixel_index = j * width + i;
            
            Color pixel = framebuffer[pixel_index] * scale;
            pixel.x = std::sqrt(pixel.x);
            pixel.y = std::sqrt(pixel.y);
            pixel.z = std::sqrt(pixel.z);
            
            int r = static_cast<int>(256 * std::clamp(pixel.x, 0.0f, 0.999f));
            int g = static_cast<int>(256 * std::clamp(pixel.y, 0.0f, 0.999f));
            int b = static_cast<int>(256 * std::clamp(pixel.z, 0.0f, 0.999f));
            
            outfile << r << ' ' << g << ' ' << b << '\n';
        }
    }
    
    outfile.close();
    std::cout << "Image saved to " << filename << std::endl;
}

std::string generate_timestamp_filename(const std::string& prefix, const std::string& extension) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << prefix << "_" << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") << "." << extension;
    return oss.str();
}

void save_image_png(
    const std::string& filename,
    const std::vector<Color>& framebuffer,
    int width,
    int height,
    int samples_per_pixel
) {
    std::string ppm_filename = filename;
    size_t extension_pos = ppm_filename.find_last_of('.');
    if (extension_pos != std::string::npos) {
        ppm_filename = ppm_filename.substr(0, extension_pos) + ".ppm";
    } else {
        ppm_filename += ".ppm";
    }
    
    save_image_ppm(ppm_filename, framebuffer, width, height, samples_per_pixel);
    
    std::cout << "PNG support not implemented. Image saved as PPM instead." << std::endl;
}
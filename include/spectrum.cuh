#ifndef SPECTRUM_CUH
#define SPECTRUM_CUH

#include "vec3.cuh"
#include <cuda_runtime.h>

constexpr float LAMBDA_MIN = 380.0f;  
constexpr float LAMBDA_MAX = 780.0f;

struct SpectralSampling {
    int num_samples;
    float lambda_min;
    float lambda_max;
    float delta_lambda;
};

__host__ __device__ inline Color wavelength_to_rgb(float wavelength) {
    float gamma = 0.8f;
    float intensity_max = 1.0f;
    float factor;
    Color rgb(0, 0, 0);

    if ((wavelength >= 380) && (wavelength < 440)) {
        rgb.x = -(wavelength - 440) / (440 - 380);
        rgb.y = 0.0f;
        rgb.z = 1.0f;
    } else if ((wavelength >= 440) && (wavelength < 490)) {
        rgb.x = 0.0f;
        rgb.y = (wavelength - 440) / (490 - 440);
        rgb.z = 1.0f;
    } else if ((wavelength >= 490) && (wavelength < 510)) {
        rgb.x = 0.0f;
        rgb.y = 1.0f;
        rgb.z = -(wavelength - 510) / (510 - 490);
    } else if ((wavelength >= 510) && (wavelength < 580)) {
        rgb.x = (wavelength - 510) / (580 - 510);
        rgb.y = 1.0f;
        rgb.z = 0.0f;
    } else if ((wavelength >= 580) && (wavelength < 645)) {
        rgb.x = 1.0f;
        rgb.y = -(wavelength - 645) / (645 - 580);
        rgb.z = 0.0f;
    } else if ((wavelength >= 645) && (wavelength < 781)) {
        rgb.x = 1.0f;
        rgb.y = 0.0f;
        rgb.z = 0.0f;
    } else {
        rgb.x = 0.0f;
        rgb.y = 0.0f;
        rgb.z = 0.0f;
    }

    if ((wavelength >= 380) && (wavelength < 420)) {
        factor = 0.3f + 0.7f * (wavelength - 380) / (420 - 380);
    } else if ((wavelength >= 420) && (wavelength < 701)) {
        factor = 1.0f;
    } else if ((wavelength >= 701) && (wavelength < 781)) {
        factor = 0.3f + 0.7f * (780 - wavelength) / (780 - 700);
    } else {
        factor = 0.0f;
    }

    rgb.x = powf(rgb.x * factor * intensity_max, gamma);
    rgb.y = powf(rgb.y * factor * intensity_max, gamma);
    rgb.z = powf(rgb.z * factor * intensity_max, gamma);

    return rgb;
}

__host__ __device__ inline float cauchy_dispersion(float wavelength, float A, float B, float C) {
    float lambda_squared = wavelength * wavelength;
    return A + B / lambda_squared + C / (lambda_squared * lambda_squared);
}

__host__ __device__ inline float glass_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.5168f, 4320.0f, 0.0f);
}

__host__ __device__ inline float flint_glass_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.5837f, 10800.0f, 0.0f);
}

__host__ __device__ inline float diamond_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 2.38f, 19470.0f, 0.0f);
}

__host__ __device__ inline float water_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.319f, 3370.0f, 0.0f);
}

#endif // SPECTRUM_CUH
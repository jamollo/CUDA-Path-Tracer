#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>
#include "generated/metal_presets.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    struct Base {
        float weight;
        glm::vec3 color;
        float diffRoughness;
    } base;

    struct Emission {
        float emittance;
    } emission;

    struct Specular {
        // a non-zero weight implies reflectiveness
        float weight;
        float roughness;
        float indexOfRefraction;
    } specular;

    struct Metallic {
        float weight;
        int metallicPresetID; // 0 = Gold, 
                              // 1 = Silver, 
                              // 2 = Copper, 
                              // 3 = Brushed Metal, 
                              // 4 = Chrome
                              
        // user-hidden derived values
        glm::vec3 etaT; // Real part of the metal's IOR
        glm::vec3 k;   // Imaginary part; nonnegative, can exceed 1

        void setComplexIOR(int metallicPresetID)
        {
            switch (metallicPresetID)
            {
            case 0:
                // set etaT & k for Gold
                break;
            case 1:
                // set etaT & k for Silver
                break;
            case 2:
                // set etaT & k for Copper
                break;
            case 3:
                // set etaT & k for Brushed Metal
                break;
            case 4:
                // set etaT & k for Chrome
                break;
            }
        }
    } metallic;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

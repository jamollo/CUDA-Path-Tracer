#pragma once

#include "sceneStructs.h"
#include <glm/glm.hpp>
#include <thrust/random.h>

// Cosine-weighted sampling about a unit world-space normal.
__host__ __device__ glm::vec3 sampleCosineWeightedHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng);

namespace bsdf {

// Local convention: normal = +Z, both directions point away from the hit,
// and reflection is supported only for wo.z > 0 and wi.z > 0.
// frontFace records the ORIGINAL outward geometric normal's side; orienting
// the reflection frame must not erase which dielectric medium is incident.
struct Sample {
    glm::vec3 wi;
    glm::vec3 f; //  material BRDF, not just the sampled component.
    float pdf;  //  mixture PDF, measured per unit solid angle.
};

__host__ __device__ glm::vec3 evaluate(
    const Material& material, const glm::vec3& wo, const glm::vec3& wi,
    bool frontFace);

__host__ __device__ float pdf(
    const Material& material, const glm::vec3& wo, const glm::vec3& wi);

// pdf == 0 denotes a zero-contribution event. dont retry that sample.
__host__ __device__ Sample sample(
    const Material& material, const glm::vec3& wo, bool frontFace,
    thrust::default_random_engine& rng);

} // namespace bsdf

// 'normal' is the outward geometric normal. Builds an oriented frame,
// updates throughput with f * cos / pdf, and cretes one offset continuation.
// An absorbing, invalid, or exhausted path is terminated with black color.
// Emission is handled BEFORE calling this function, in shadeMaterial.
__host__ __device__ void scatterRay(
    PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal,
    const Material& material, thrust::default_random_engine& rng);

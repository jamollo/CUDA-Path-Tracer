#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cfloat>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

struct IsActive {
    __host__ __device__
        bool operator()(const PathSegment& path) const {
        return path.remainingBounces > 0;
    }
};

struct CompareMaterial {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a,
            const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    // Hash each dimension without signed shifts or overlapping bit fields.
    unsigned int seed = utilhash(static_cast<unsigned int>(iter));
    seed = utilhash(seed ^ static_cast<unsigned int>(index));
    seed = utilhash(seed ^ static_cast<unsigned int>(depth));
    return thrust::default_random_engine(seed == 0u ? 1u : seed);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));


    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    dev_image = nullptr;
    dev_paths = nullptr;
    dev_geoms = nullptr;
    dev_materials = nullptr;
    dev_intersections = nullptr;

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = traceDepth > 0 ? glm::vec3(1.0f) : glm::vec3(0.0f);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        const float sampleX = static_cast<float>(x) + u01(rng);
        const float sampleY = static_cast<float>(y) + u01(rng);

        // One uniform subpixel sample per pixel per iteration.
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// Every active path gets a complete hit record, including initialized misses.
__global__ void computeIntersections(
    int depth, int num_paths, PathSegment* pathSegments,
    Geom* geoms, int geoms_size, ShadeableIntersection* intersections)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;
    const Ray ray = pathSegments[idx].ray;
    ShadeableIntersection best{-1.0f, glm::vec3(0.0f), -1, glm::vec3(0.0f)};
    float closest = FLT_MAX;
    for (int i = 0; i < geoms_size; ++i) {
        glm::vec3 position(0.0f), normal(0.0f);
        bool outside = true;
        float t = -1.0f;
        if (geoms[i].type == CUBE) {
            t = boxIntersectionTest(geoms[i], ray, position, normal, outside);
        }
        else if (geoms[i].type == SPHERE) {
            t = sphereIntersectionTest(geoms[i], ray, position, normal, outside);
        }
        if (t > 0.0f && t < closest) {
            closest = t;
            best.t = t;
            best.materialId = geoms[i].materialid;
            best.surfaceNormal = normal; // Still outward, never face-forwarded.
            best.position = position;    // Preserve the exact intersection output.
        }
    }
    intersections[idx] = best;
}

__global__ void shadeMaterial(
    int iter, int num_paths, ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments, Material* materials)
{
    int idx;
    if (!utilityCore::indexIsValid(num_paths, idx)) return;
    PathSegment& path = pathSegments[idx];
    if (path.remainingBounces <= 0) return;
    const ShadeableIntersection& hit = shadeableIntersections[idx];
    if (hit.t <= 0.0f) {
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
        return;
    }

    const Material& material = materials[hit.materialId];
    const glm::vec3 emitted = material.emission.color * material.emission.emittance;
    if (emitted.x > 0.0f || emitted.y > 0.0f || emitted.z > 0.0f) {
        // This path's throughput becomes its final radiance contribution.
        // Check emission before bounce exhaustion, including the last query.
        path.color *= emitted;
        path.remainingBounces = 0;
        return;
    }

    // Seed by ORIGINAL pixel identity, never the sorted/compacted array slot.
    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, path.pixelIndex, path.remainingBounces);
    scatterRay(path, hit.position, hit.surfaceNormal, material, rng);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = traceDepth > 0 ? pixelcount : 0;
    if (guiData != nullptr) guiData->TracedDepth = 0;

    while (num_paths > 0) {
        const int blocks = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<blocks, blockSize1d>>>(
            depth, num_paths, dev_paths, dev_geoms,
            static_cast<int>(hst_scene->geoms.size()), dev_intersections);
        checkCUDAError("trace one bounce");
        ++depth;

        if (hst_scene->state.sortMaterials) {
            // Sort hit keys AND corresponding paths together, only in the
            // active prefix. Material-array indices remain unchanged.
            thrust::sort_by_key(thrust::device,
                dev_intersections, dev_intersections + num_paths,
                dev_paths, CompareMaterial{});
        }

        shadeMaterial<<<blocks, blockSize1d>>>(
            iter, num_paths, dev_intersections, dev_paths, dev_materials);
        checkCUDAError("shade one bounce");

        // Partition preserves both groups. Finished contributions accumulate
        // in the tail; previously finished paths beyond this prefix stay put.
        // remove_if would NOT preserve that tail for the final full-array gather.
        PathSegment* activeEnd = thrust::partition(thrust::device,
            dev_paths, dev_paths + num_paths, IsActive{});
        num_paths = static_cast<int>(activeEnd - dev_paths);
        if (guiData != nullptr) guiData->TracedDepth = depth;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

    finalGather<<<numBlocksPixels, blockSize1d>>>(
        pixelcount,
        dev_image,
        dev_paths
        );

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

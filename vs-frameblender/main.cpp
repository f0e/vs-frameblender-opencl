#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>
#include <CL/opencl.hpp>
#include <iostream>
#include "../vapoursynth/VSHelper4.h"
#include "../vapoursynth/VapourSynth4.h"

#define RETERROR(x) do { vsapi->mapSetError(out, (x)); return; } while (0)

const char* kernel_source = R"(
__kernel void frame_blend_kernel(__global const float *weights,
                                 __global const uchar *srcs,
                                 int num_srcs,
                                 __global uchar *dst,
                                 int depth,
                                 int width,
                                 int height,
                                 int stride) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height)
        return;
    
    float acc = 0.0f;
    int maxval = (1 << depth) - 1;
    int dst_offset = y * stride + x;
    
    for (int i = 0; i < num_srcs; ++i) {
        int src_offset = i * (height * stride) + dst_offset;
        acc += convert_float(srcs[src_offset]) * weights[i];
    }
    
    int result = convert_int_sat_rte(acc);
    dst[dst_offset] = convert_uchar_sat(clamp(result, 0, maxval));
}
)";

typedef struct {
    VSNode* node;
    const VSVideoInfo* vi;

    std::vector<float> weightPercents;
    bool process[3];

    cl::Context context;
    std::unique_ptr<cl::Kernel> kernel;
    cl::CommandQueue queue;
} FrameBlendData;

static void frameBlend(VSCore* core, const FrameBlendData* d, const VSFrame* const* srcs, VSFrame* dst, int plane, const VSAPI* vsapi) {
    int stride = vsapi->getStride(dst, plane);
    int width = vsapi->getFrameWidth(dst, plane);
    int height = vsapi->getFrameHeight(dst, plane);
    int depth = d->vi->format.bitsPerSample;
    const int num_frames = static_cast<int>(d->weightPercents.size());

    size_t frame_size = stride * height;
    size_t total_src_size = frame_size * num_frames;

    // Create OpenCL buffers
    cl::Buffer weights_buf(d->context, CL_MEM_READ_ONLY, num_frames * sizeof(float));
    cl::Buffer srcs_buf(d->context, CL_MEM_READ_ONLY, total_src_size);
    cl::Buffer dst_buf(d->context, CL_MEM_WRITE_ONLY, frame_size);

    // Copy weights to the OpenCL buffer
    d->queue.enqueueWriteBuffer(weights_buf, CL_FALSE, 0, num_frames * sizeof(float), d->weightPercents.data());

    // Copy source data to the OpenCL buffer
    for (int i = 0; i < num_frames; ++i) {
        const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(srcs[i], plane));
        d->queue.enqueueWriteBuffer(srcs_buf, CL_FALSE, i * frame_size, frame_size, src_ptr);

        d->queue.enqueueWriteBuffer(weights_buf, CL_FALSE, i * sizeof(float), sizeof(float), &d->weightPercents[i]);
    }

    // Set kernel arguments
    d->kernel->setArg(0, weights_buf);
    d->kernel->setArg(1, srcs_buf);
    d->kernel->setArg(2, num_frames);
    d->kernel->setArg(3, dst_buf);
    d->kernel->setArg(4, depth);
    d->kernel->setArg(5, width);
    d->kernel->setArg(6, height);
    d->kernel->setArg(7, stride);

    // Execute the kernel
    d->queue.enqueueNDRangeKernel(*d->kernel, cl::NullRange, cl::NDRange(width, height));

    // Read the result back into the destination frame
    uint8_t* dstp = reinterpret_cast<uint8_t*>(vsapi->getWritePtr(dst, plane));
    d->queue.enqueueReadBuffer(dst_buf, CL_TRUE, 0, frame_size, dstp);

    // Finish all pending operations
    d->queue.finish();
}

template<typename T>
static void frameBlendOld(const FrameBlendData* d, const VSFrame* const* srcs, VSFrame* dst, int plane, const VSAPI* vsapi) {
    int stride = vsapi->getStride(dst, plane) / sizeof(T);
    int width = vsapi->getFrameWidth(dst, plane);
    int height = vsapi->getFrameHeight(dst, plane);

    const T* srcpp[128];
    const size_t numSrcs = d->weightPercents.size();

    for (size_t i = 0; i < numSrcs; i++) {
        srcpp[i] = reinterpret_cast<const T*>(vsapi->getReadPtr(srcs[i], plane));
    }

    T* VS_RESTRICT dstp = reinterpret_cast<T*>(vsapi->getWritePtr(dst, plane));

    unsigned maxVal = (1U << d->vi->format.bitsPerSample) - 1;

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            float acc = 0;

            for (size_t i = 0; i < numSrcs; ++i) {
                T val = srcpp[i][w];
                acc += val * d->weightPercents[i];
            }

            int actualAcc = std::clamp(int(acc), 0, int(maxVal));
            dstp[w] = static_cast<T>(actualAcc);
        }

        for (size_t i = 0; i < numSrcs; ++i) {
            srcpp[i] += stride;
        }
        dstp += stride;
    }
}

static const VSFrame* VS_CC frameBlendGetFrame(
    int n, int activationReason, void* instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core,
    const VSAPI* vsapi
) {
    FrameBlendData* d = static_cast<FrameBlendData*>(instanceData);

    const int half = int(d->weightPercents.size() / 2);

    if (activationReason == arInitial) {
        bool clamp = (n > INT_MAX - 1 - half);
        int lastframe = clamp ? INT_MAX - 1 : n + half;

        // request all the frames we'll need
        for (int i = std::max(0, n - half); i <= lastframe; i++) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
    }
    else if (activationReason == arAllFramesReady) {
        // get this frame's frames to be blended
        std::vector<const VSFrame*> frames(d->weightPercents.size());

        int fn = n - half;
        for (size_t i = 0; i < d->weightPercents.size(); i++) {
            frames[i] = vsapi->getFrameFilter(std::max(0, fn), d->node, frameCtx);
            if (fn < INT_MAX - 1) fn++;
        }

        const VSFrame* center = frames[frames.size() / 2];
        const VSVideoFormat* fi = vsapi->getVideoFrameFormat(center);

        const int pl[] = { 0, 1, 2 };
        const VSFrame* fr[] = {
            d->process[0] ? nullptr : center,
            d->process[1] ? nullptr : center,
            d->process[2] ? nullptr : center
        };

        VSFrame* dst;
        dst = vsapi->newVideoFrame2(
            fi, vsapi->getFrameWidth(center, 0), vsapi->getFrameHeight(center, 0), fr, pl, center, core
        );

        for (int plane = 0; plane < fi->numPlanes; plane++) {
            if (d->process[plane]) {
                if (fi->bytesPerSample == 1) {
                    frameBlend(core, d, frames.data(), dst, plane, vsapi);
                }
                else {
                    return nullptr;
                }
            }
        }

        for (auto iter : frames)
            vsapi->freeFrame(iter);

        return dst;
    }

    return nullptr;
}

static void VS_CC frameBlendFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    FrameBlendData* d = static_cast<FrameBlendData*>(instanceData);

    vsapi->freeNode(d->node);

    free(d);
}

static void VS_CC frameBlendCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    std::unique_ptr<FrameBlendData> data(new FrameBlendData());
    int err;

    int numWeights = vsapi->mapNumElements(in, "weights");
    if ((numWeights % 2) != 1)
        RETERROR("FrameBlend: Number of weights must be odd");

    // get clip and clip video info
    data->node = vsapi->mapGetNode(in, "clip", 0, &err);
    if (err) {
        vsapi->freeNode(data->node);
        RETERROR("FrameBlend: clip is not a valid VideoNode");
    }

    data->vi = vsapi->getVideoInfo(data->node);

    // get weights
    float totalWeights = 0.f;
    for (int i = 0; i < numWeights; i++)
        totalWeights += vsapi->mapGetFloat(in, "weights", i, 0);

    // scale weights
    for (int i = 0; i < numWeights; i++) {
        data->weightPercents.push_back(vsapi->mapGetFloat(in, "weights", i, 0) / totalWeights);
    }

    int nPlanes = vsapi->mapNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        data->process[i] = (nPlanes <= 0); // default to all planes if no planes specified

    if (nPlanes <= 0) nPlanes = 3;

    for (int i = 0; i < nPlanes; i++) {
        int plane = vsapi->mapGetInt(in, "planes", i, &err);

        if (plane < 0 || plane >= 3) {
            vsapi->freeNode(data->node);
            RETERROR("FrameBlend: plane index out of range");
        }

        data->process[plane] = true;
    }

    // Set up OpenCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        vsapi->freeNode(data->node);
        RETERROR("FrameBlend: No OpenCL platforms found");
    }
    
    data->context = cl::Context(CL_DEVICE_TYPE_GPU);
    std::vector<cl::Device> devices = data->context.getInfo<CL_CONTEXT_DEVICES>();
    
    data->queue = cl::CommandQueue(data->context, devices[0]);
    
    // Compile the OpenCL kernel
    cl::Program::Sources sources = { kernel_source };
    cl::Program program(data->context, sources);
    if (program.build(devices) != CL_SUCCESS) {
        vsapi->freeNode(data->node);
        RETERROR("FrameBlend: Error building OpenCL program");
    }
    
    data->kernel = std::make_unique<cl::Kernel>(cl::Kernel(program, "frame_blend_kernel"));

    VSFilterDependency deps[] = {
        {data->node, rpGeneral}
    };

    vsapi->createVideoFilter(
        out, "FrameBlend", data->vi, frameBlendGetFrame, frameBlendFree, fmParallelRequests, deps, 1, data.get(), core
    );

    data.release();
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin(
        "com.github.animafps.vs-frameblender-opencl", "frameblenderopencl", "Frame blender", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION,
        0, plugin
    );
    vspapi->registerFunction(
        "FrameBlend", "clip:vnode;weights:float[];planes:int[]:opt;", "clip:vnode;", frameBlendCreate, NULL, plugin
    );
}
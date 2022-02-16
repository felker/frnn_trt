#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #define HAS_LOGGING

static bool file_exists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

static std::string dims_info(const nvinfer1::Dims& dims)
{
    std::stringstream ss;
    ss << "ndims=" << dims.nbDims << "; [";
    for (int i = 0; i < dims.nbDims; i++)
        ss << " " << dims.d[i];
    ss << " ]";
    return ss.str();
}

static std::size_t dims_element_count(const nvinfer1::Dims& dims)
{
    std::size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        count *= dims.d[i];
    }
    return count;
}

static std::size_t dtype_size(const nvinfer1::DataType& dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    default:
        throw std::runtime_error("unknown dtype");
    }
}

namespace infer {

struct NvInferDeleter final
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
using shared_ptr = std::shared_ptr<T>;

template <typename T>
using unique_ptr = std::unique_ptr<T, NvInferDeleter>;

template <typename T>
std::shared_ptr<T> make_shared(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, NvInferDeleter());
};

template <typename T>
unique_ptr<T> make_unique(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::unique_ptr<T, NvInferDeleter>(obj);
}

class Logger final : public nvinfer1::ILogger
{
  public:
    ~Logger() final = default;
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept final
    {
#ifdef HAS_LOGGING
        std::cerr << msg << std::endl;
#else
#endif
    }
};

class Runtime final
{
  public:
    Runtime() = delete;
    Runtime(std::unique_ptr<nvinfer1::ILogger> logger) :
      m_logger(std::move(logger)),
      m_runtime(make_unique(nvinfer1::createInferRuntime(*m_logger)))
    {}

    nvinfer1::IRuntime& unwrap()
    {
        return *m_runtime;
    }

  private:
    std::unique_ptr<nvinfer1::ILogger> m_logger;
    unique_ptr<nvinfer1::IRuntime> m_runtime;
};

class Engine final
{
  public:
    Engine() = delete;
    Engine(std::string plan_file, std::shared_ptr<Runtime> runtime) : m_runtime(std::move(runtime))
    {
        if (!file_exists(plan_file))
        {
            throw std::runtime_error("tensorrt engine file does not exist: " + plan_file);
        }

        std::ifstream file(plan_file, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        if (size <= 0)
        {
            throw std::runtime_error("tensorrt engine file is empty");
        }

        std::vector<char> buffer;
        buffer.reserve(size);

        file.seekg(0, std::ios::beg);
        try
        {
            file.read(buffer.data(), size);
        } catch (...)
        {
            throw std::runtime_error("error reading tensorrt engine file");
        }
        file.close();

        m_engine = make_shared(m_runtime->unwrap().deserializeCudaEngine(buffer.data(), size, nullptr));
    }

    nvinfer1::ICudaEngine& unwrap()
    {
        return *m_engine;
    }

  private:
    std::shared_ptr<Runtime> m_runtime;
    shared_ptr<nvinfer1::ICudaEngine> m_engine;
};

class ExecutionContext final
{
  public:
    ExecutionContext() = delete;
    ExecutionContext(std::shared_ptr<Engine> engine) :
      m_engine(std::move(engine)),
      m_context(make_unique(m_engine->unwrap().createExecutionContext()))
    {}

    nvinfer1::IExecutionContext& unwrap()
    {
        return *m_context;
    }

    std::size_t binding_size_in_bytes(std::uint32_t binding_id)
    {
        auto dims  = m_context->getBindingDimensions(binding_id);
        auto dtype = m_context->getEngine().getBindingDataType(binding_id);
        return dims_element_count(dims) * dtype_size(dtype);
    }

  private:
    std::shared_ptr<Engine> m_engine;
    unique_ptr<nvinfer1::IExecutionContext> m_context;
};

}  // namespace infer

#define CHECK(cond, str)                   \
    {                                      \
        if (not(cond))                     \
            throw std::runtime_error(str); \
    }

#define CHECK_CUDA(cond, str)                                                      \
    {                                                                              \
        auto status = (cond);                                                      \
        if (status != cudaSuccess)                                                 \
        {                                                                          \
            std::string msg = std::string(str) + " " + cudaGetErrorString(status); \
            throw std::runtime_error(msg);                                         \
        }                                                                          \
    }

int main(int argc, char* argv[])
{
    const std::size_t warmup_iters = 10;
    const std::size_t timing_iters = 1000;

    // set up tensorrt components

    auto logger  = std::make_unique<infer::Logger>();
    auto runtime = std::make_shared<infer::Runtime>(std::move(logger));
    auto engine  = std::make_shared<infer::Engine>("./frnn.engine", runtime);
    auto context = std::make_shared<infer::ExecutionContext>(engine);

    // you will want to generalize bindings, but since this model has one input and one output
    // we will allocation buffers specific to the engine in question

    auto nbindings = engine->unwrap().getNbBindings();
    CHECK(nbindings == 2, "incorrect binding count");

    CHECK(engine->unwrap().bindingIsInput(0), "expected binding 0 to be the input binding");
    CHECK(!engine->unwrap().bindingIsInput(1), "expected binding 1 to be the output binding");

    auto input_dims  = engine->unwrap().getBindingDimensions(0);
    auto output_dims = engine->unwrap().getBindingDimensions(1);

    std::cout << "input bindings: " << dims_info(input_dims) << "; bytes: " << context->binding_size_in_bytes(0)
              << std::endl;
    std::cout << "output bindings: " << dims_info(output_dims) << "; bytes: " << context->binding_size_in_bytes(1)
              << std::endl;

    // set up cuda components

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream), "failed to create CUDA stream");

    void* input_h           = nullptr;
    void* input_d           = nullptr;
    std::size_t input_bytes = context->binding_size_in_bytes(0);

    void* output_h           = nullptr;
    void* output_d           = nullptr;
    std::size_t output_bytes = context->binding_size_in_bytes(1);

    // allocate input tensors

    CHECK_CUDA(cudaMallocHost(&input_h, input_bytes), "failed to allocate pinned memory");
    CHECK_CUDA(cudaMalloc(&input_d, input_bytes), "failed to allocate device memory");

    // allocate output tensors

    CHECK_CUDA(cudaMallocHost(&output_h, output_bytes), "failed to allocate pinned memory");
    CHECK_CUDA(cudaMalloc(&output_d, output_bytes), "failed to allocate device memory");

    // touch the memory

    std::memset(input_h, 0, input_bytes);
    std::memset(output_h, 0, output_bytes);
    CHECK_CUDA(cudaMemset(input_d, 0, input_bytes), "cuda memset failed");
    CHECK_CUDA(cudaMemset(output_d, 0, output_bytes), "cuda memset failed");

    // execution context device bindings

    std::array<void*, 2> bindings = {input_d, output_d};

    CHECK_CUDA(cudaStreamSynchronize(stream), "stream sync failed");
    for (int i = 0; i < warmup_iters; ++i)
    {
        CHECK_CUDA(cudaMemcpyAsync(input_d, input_h, input_bytes, cudaMemcpyHostToDevice, stream), "h2d failed");
        context->unwrap().enqueue(1, bindings.data(), stream, nullptr);
        CHECK_CUDA(cudaMemcpyAsync(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost, stream), "d2h failed");
        CHECK_CUDA(cudaStreamSynchronize(stream), "stream sync failed");
    }
    CHECK_CUDA(cudaStreamSynchronize(stream), "stream sync failed");

    CHECK_CUDA(cudaStreamSynchronize(stream), "stream sync failed");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timing_iters; ++i)
    {
        CHECK_CUDA(cudaMemcpyAsync(input_d, input_h, input_bytes, cudaMemcpyHostToDevice, stream), "h2d failed");
        context->unwrap().enqueue(1, bindings.data(), stream, nullptr);
        CHECK_CUDA(cudaMemcpyAsync(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost, stream), "d2h failed");
        CHECK_CUDA(cudaStreamSynchronize(stream), "stream sync failed");
    }
    CHECK_CUDA(cudaStreamSynchronize(stream), "stream sync failed");
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "timing for " << timing_iters << " iterations: " << elapsed << " seconds" << std::endl;

    // free memory
    CHECK_CUDA(cudaFreeHost(input_h), "cudaFreeHost error");
    CHECK_CUDA(cudaFreeHost(output_h), "cudaFreeHost error");
    CHECK_CUDA(cudaFree(input_d), "cudaFree error");
    CHECK_CUDA(cudaFree(output_d), "cudaFree error");

    // final sync
    CHECK_CUDA(cudaStreamSynchronize(stream), "stream sync failed");
    CHECK_CUDA(cudaStreamDestroy(stream), "stream destroy failed");

    // tensorrt component will properly destruct in the right sequence
    // based on the shared_ptr reference count
}

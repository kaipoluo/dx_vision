#include "trt_engine.hpp"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "tools/logger.hpp"

namespace auto_aim
{

// --- RAII wrappers for TensorRT C-API objects ---

static void checkCuda(cudaError_t err, const char * tag)
{
  if (err != cudaSuccess) {
    auto msg = std::string(tag) + ": " + cudaGetErrorString(err);
    tools::logger()->error("{}", msg);
    throw std::runtime_error(msg);
  }
}

namespace
{

class Logger : public nvinfer1::ILogger
{
public:
  static Logger & instance()
  {
    static Logger l;
    return l;
  }

  void log(Severity severity, const char * msg) noexcept override
  {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        tools::logger()->error("[TRT] {}", msg);
        break;
      case Severity::kWARNING:
        tools::logger()->warn("[TRT] {}", msg);
        break;
      case Severity::kINFO:
        tools::logger()->info("[TRT] {}", msg);
        break;
      default:
        break;
    }
  }
};

}  // namespace

// ==================== public ====================

TrtEngine::TrtEngine(
  const std::string & onnx_path, const std::string & engine_path, bool fp16)
: engine_path_(engine_path)
{
  // 尝试加载缓存的 engine，若不存在则从 ONNX 构建
  std::ifstream f(engine_path, std::ios::binary);
  if (f.good()) {
    f.close();
    tools::logger()->info("[TrtEngine] loading cached engine: {}", engine_path);
    loadEngine(engine_path);
  } else {
    tools::logger()->info("[TrtEngine] building engine from ONNX: {}", onnx_path);
    buildFromOnnx(onnx_path, engine_path, fp16);
  }

  // 确定 I/O 规格
  auto input_dims = engine_->getTensorShape(input_name_.c_str());
  width_ = input_dims.d[3];
  height_ = input_dims.d[2];
  input_size_ = input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3] *
                sizeof(float);

  auto output_dims = engine_->getTensorShape(output_name_.c_str());
  output_size_ = 1;
  for (int i = 0; i < output_dims.nbDims; ++i) {
    output_size_ *= output_dims.d[i];
  }
  output_size_ *= sizeof(float);

  // 创建 execution context
  context_ = engine_->createExecutionContext();
  if (!context_) {
    throw std::runtime_error("Failed to create execution context");
  }

  // 创建 CUDA stream
  checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

  // 分配 GPU buffer
  checkCuda(cudaMalloc(&device_input_, input_size_), "cudaMalloc input");
  checkCuda(cudaMalloc(&device_output_, output_size_), "cudaMalloc output");

  // 绑定到 context
  context_->setInputTensorAddress(input_name_.c_str(), device_input_);
  context_->setOutputTensorAddress(output_name_.c_str(), device_output_);

  tools::logger()->info(
    "[TrtEngine] ready — input {}x{} ({} MB), output {} MB", width_, height_,
    input_size_ / 1024 / 1024, output_size_ / 1024 / 1024);
}

TrtEngine::~TrtEngine()
{
  if (stream_) cudaStreamDestroy(stream_);
  if (device_input_) cudaFree(device_input_);
  if (device_output_) cudaFree(device_output_);
  // context_ and engine_ are owned by the unique_ptr / raw pointer aliases
  // engine_ is owned by runtime_; context_ standalone
  delete context_;
  // runtime_ reset will destroy engine_
}

TrtEngine::TrtEngine(TrtEngine && other) noexcept
: engine_path_(std::move(other.engine_path_)),
  width_(other.width_),
  height_(other.height_),
  input_size_(other.input_size_),
  output_size_(other.output_size_),
  runtime_(std::move(other.runtime_)),
  engine_(other.engine_),
  context_(other.context_),
  stream_(other.stream_),
  device_input_(other.device_input_),
  device_output_(other.device_output_),
  input_name_(std::move(other.input_name_)),
  output_name_(std::move(other.output_name_))
{
  other.engine_ = nullptr;
  other.context_ = nullptr;
  other.stream_ = nullptr;
  other.device_input_ = nullptr;
  other.device_output_ = nullptr;
}

TrtEngine & TrtEngine::operator=(TrtEngine && other) noexcept
{
  if (this != &other) {
    this->~TrtEngine();
    engine_path_ = std::move(other.engine_path_);
    width_ = other.width_;
    height_ = other.height_;
    input_size_ = other.input_size_;
    output_size_ = other.output_size_;
    runtime_ = std::move(other.runtime_);
    engine_ = other.engine_;
    context_ = other.context_;
    stream_ = other.stream_;
    device_input_ = other.device_input_;
    device_output_ = other.device_output_;
    input_name_ = std::move(other.input_name_);
    output_name_ = std::move(other.output_name_);

    other.engine_ = nullptr;
    other.context_ = nullptr;
    other.stream_ = nullptr;
    other.device_input_ = nullptr;
    other.device_output_ = nullptr;
  }
  return *this;
}

bool TrtEngine::infer(const float * input, float * output)
{
  // H2D
  checkCuda(
    cudaMemcpyAsync(
      device_input_, input, input_size_, cudaMemcpyHostToDevice, stream_),
    "H2D");

  // enqueue inference
  auto ok = context_->enqueueV3(stream_);
  if (!ok) {
    tools::logger()->error("[TrtEngine] enqueueV3 failed");
    return false;
  }

  // D2H
  checkCuda(
    cudaMemcpyAsync(
      device_output_, output, output_size_, cudaMemcpyDeviceToHost, stream_),
    "D2H");

  // sync — 确保数据已经拷回
  checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return true;
}

// ==================== private ====================

void TrtEngine::buildFromOnnx(
  const std::string & onnx_path, const std::string & engine_path, bool fp16)
{
  // 注册 plugin 支持
  initLibNvInferPlugins(&Logger::instance(), "");

  // Builder
  auto builder =
    std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(Logger::instance()));
  if (!builder) throw std::runtime_error("createInferBuilder failed");

  auto network =
    std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (!network) throw std::runtime_error("createNetworkV2 failed");

  auto config =
    std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) throw std::runtime_error("createBuilderConfig failed");

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1 GB

  if (fp16 && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    tools::logger()->info("[TrtEngine] FP16 enabled");
  }

  // ONNX parser
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, Logger::instance()));
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("Failed to parse ONNX: " + onnx_path);
  }

  // 记下 I/O tensor 名称（在 build 前，network 还活着）
  {
    int n_inputs = network->getNbInputs();
    if (n_inputs != 1) {
      throw std::runtime_error("Expected exactly 1 input, got " + std::to_string(n_inputs));
    }
    input_name_ = network->getInput(0)->getName();

    int n_outputs = network->getNbOutputs();
    if (n_outputs != 1) {
      throw std::runtime_error("Expected exactly 1 output, got " + std::to_string(n_outputs));
    }
    output_name_ = network->getOutput(0)->getName();

    tools::logger()->info("[TrtEngine] input: '{}', output: '{}'", input_name_, output_name_);
  }

  // Build serialized engine
  auto serialized =
    std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!serialized) throw std::runtime_error("buildSerializedNetwork failed");

  // 缓存到磁盘
  {
    std::ofstream out(engine_path, std::ios::binary);
    out.write(static_cast<const char *>(serialized->data()), serialized->size());
    tools::logger()->info(
      "[TrtEngine] engine cached to: {} ({} MB)", engine_path, serialized->size() / 1024 / 1024);
  }

  // 加载 memory 中的 engine
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(Logger::instance()));
  engine_ = deserializeEngine(*runtime_, std::vector<uint8_t>(
    static_cast<const uint8_t *>(serialized->data()),
    static_cast<const uint8_t *>(serialized->data()) + serialized->size()));
}

void TrtEngine::loadEngine(const std::string & engine_path)
{
  // 读磁盘
  std::ifstream in(engine_path, std::ios::binary | std::ios::ate);
  auto size = in.tellg();
  in.seekg(0, std::ios::beg);

  std::vector<uint8_t> data(static_cast<size_t>(size));
  in.read(reinterpret_cast<char *>(data.data()), size);
  in.close();

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(Logger::instance()));
  engine_ = deserializeEngine(*runtime_, data);

  // 恢复 I/O 名称
  {
    int n_inputs = engine_->getNbIOTensors();
    for (int i = 0; i < n_inputs; ++i) {
      auto name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        input_name_ = name;
        break;
      }
    }
    for (int i = 0; i < n_inputs; ++i) {
      auto name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        output_name_ = name;
        break;
      }
    }
    tools::logger()->info(
      "[TrtEngine] loaded engine I/O: '{}' -> '{}'", input_name_, output_name_);
  }
}

nvinfer1::ICudaEngine * TrtEngine::deserializeEngine(
  nvinfer1::IRuntime & runtime, const std::vector<uint8_t> & data)
{
  auto * engine = runtime.deserializeCudaEngine(data.data(), data.size());
  if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
  return engine;
}

}  // namespace auto_aim
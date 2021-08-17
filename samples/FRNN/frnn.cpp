/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! FRNN.cpp
//! Command: ./frnn [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <ctime>

const std::string gSampleName = "FRNN";

//! \brief  The FRNN class implements the LSTM network
//!
//! \details It creates the network using an ONNX model
//!
class FRNN
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    FRNN(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies the input signals and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the network by parsing the ONNX model and builds
//!          the engine that will be used to run inference (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool FRNN::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 3);

    sample::gLogInfo << "mInputDims.nbDims = " << mInputDims.nbDims << std::endl;

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 3);

    sample::gLogInfo << "mOutputDims.nbDims = " << mOutputDims.nbDims << std::endl;

    return true;
}

//!
//! \brief Uses a ONNX parser to create the network and marks the output layers
//!
//! \param network Pointer to the network that will be populated with the network defined
//! in the ONNX file
//!
//! \param builder Pointer to the engine builder
//!
bool FRNN::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    // KGF: despite README's instructions, this function is not implemented
    // parser->reportParsingInfo();


    config->setMaxWorkspaceSize(1024_MiB);//16_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        //samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool FRNN::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool FRNN::processInput(const samplesCommon::BufferManager& buffers)
{
  // 256, 128, 14
    const int input0 = mInputDims.d[0];
    const int input1 = mInputDims.d[1];
    const int input2 = mInputDims.d[2];

    // sample::gLogInfo << "Input dims: input0,input1,input2 = " << input0 << "," << input1 << "," << input2 << std::endl;

    // Read a random digit file
    // srand(unsigned(time(nullptr)));
    // std::vector<uint8_t> fileData(inputH * inputW);
    // mNumber = rand() % 10;
    // readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // // Print an ascii representation
    // sample::gLogInfo << "Input:" << std::endl;
    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    // }
    // sample::gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // KGF: is input0, batch size, automatically ignored when the data buffers are allocated?
    for (int i = 0; i < input1*input2; i++)
    {
      hostDataBuffer[i] = 1.0;
        // hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool FRNN::verifyOutput(const samplesCommon::BufferManager& buffers)
{
  // KGF: same question as above: is d[0] (batch size) implicitly ignored, or assumed to
  // be 1?
    const int outputSize = mOutputDims.d[1];
    // 256, 128, 1
    sample::gLogInfo << "Output dims: output0,output1,output2 = " << mOutputDims.d[0] << "," << mOutputDims.d[1] << "," << mOutputDims.d[2] << std::endl;

    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    // float val{0.0f};
    // int idx{0};

    // // Calculate Softmax
    // float sum{0.0f};
    // for (int i = 0; i < outputSize; i++)
    // {
    //     output[i] = exp(output[i]);
    //     sum += output[i];
    // }

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        // output[i] /= sum;
        // val = std::max(val, output[i]);
        // if (val == output[i])
        // {
        //     idx = i;
        // }

        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         // << " "
                         // << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
                         << std::endl;
    }

    sample::gLogInfo << std::endl;

    return true; // idx == mNumber && val > 0.9f;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("./");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "input.onnx"
    // KGF: inspect the ONNX model (e.g. with Netron) to get the names of these tensors
    params.inputTensorNames.push_back("input_2");
    params.outputTensorNames.push_back("time_distributed");
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./frnn [-h or --help] [-d or --datadir=<path to ONNX model>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       ONNX model location (and name?)"
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    FRNN sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine from ONNX file" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    clock_t begin = clock();
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    double elapsed_ms = elapsed_secs*1000;
    std::cout << std::scientific << std::setprecision(15);
    std::cout << "infer() elapsed " << elapsed_ms << " ms\n";

    return sample::gLogger.reportPass(sampleTest);
}

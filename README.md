# frnn_trt
TensorRT 8.x inference code for real-time detection of plasma disruptions

Sample code and common headers from `TensorRT-8.0.0.3.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar`

- Download appropriate tarball from https://developer.nvidia.com/nvidia-tensorrt-8x-download
- Untar
- `cd TensorRT-8.0.0.3/`
- `rm -rfd samples/`
- Untar the attached FRNN/ folder into that directory
- `make`
- `cd ../bin/`
- Copy an `.onnx` file to that directory
- `./frnn`

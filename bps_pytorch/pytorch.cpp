// Copyright (c) Meta Platforms, Inc. and affiliates.
// The modifications of this code are licensed under the CC BY-NC-SA 4.0 license found in the
// LICENSE file in the root directory of this source tree.

// The original code was copied from https://github.com/shacklettbp/bps-nav/blob/master/simulator/pytorch.cpp
// which is licensed under the MIT license.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
namespace py = pybind11;

// Create a tensor that references this memory
//
at::Tensor convertToTensorColor(const py::capsule &ptr_capsule,
                                int dev_id,
                                uint32_t batch_size,
                                const array<uint32_t, 2> &resolution)
{
    uint8_t *dev_ptr(ptr_capsule);

    array<int64_t, 4> sizes {{batch_size, resolution[0], resolution[1], 4}};

    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

// Create a tensor that references this memory
//
at::Tensor convertToTensorDepth(const py::capsule &ptr_capsule,
                                int dev_id,
                                uint32_t batch_size,
                                const array<uint32_t, 2> &resolution)
{
    float *dev_ptr(ptr_capsule);

    array<int64_t, 3> sizes {{batch_size, resolution[0], resolution[1]}};

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

// TensorRT helpers
at::Tensor convertToTensorFCOut(const py::capsule &ptr_capsule,
                                int dev_id,
                                uint32_t batch_size,
                                uint32_t num_features)
{ 
    __half *dev_ptr(ptr_capsule);

    array<int64_t, 2> sizes {{batch_size, num_features}};

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

py::capsule tensorToCapsule(const at::Tensor &tensor)
{
    return py::capsule(tensor.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("make_color_tensor", &convertToTensorColor);
    m.def("make_depth_tensor", &convertToTensorDepth);
    m.def("make_fcout_tensor", &convertToTensorFCOut);
    m.def("tensor_to_capsule", &tensorToCapsule);
}

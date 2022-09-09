// Copyright 2021 The WebNN-native Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_DMLUTILS_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_DMLUTILS_H_

#include "base/logging.h"

#define WEBNN_CHECK(hr)                   \
  if (((HRESULT)(hr)) < 0) {              \
    LOG(ERROR) << "Failed to do " << #hr; \
    assert(0);                            \
  }

namespace content::webnn {

using namespace Microsoft::WRL;
using ml::webnn::mojom::AutoPad;

inline ComPtr<ID3D12Resource> CreateCommittedResource(
    ID3D12Device* d3d12_device,
    D3D12_HEAP_TYPE heap_type,
    const D3D12_RESOURCE_DESC& resource_descriptor,
    D3D12_RESOURCE_STATES initial_usage) {
  D3D12_HEAP_PROPERTIES heap_properties;
  heap_properties.Type = heap_type;
  heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_properties.CreationNodeMask = 1;
  heap_properties.VisibleNodeMask = 1;

  // Check the resource size is valid, too large size could cause a device loss
  // when creating the resource.
  D3D12_RESOURCE_ALLOCATION_INFO resource_info =
      d3d12_device->GetResourceAllocationInfo(
          0, 1, &resource_descriptor);
  if (resource_info.SizeInBytes == 0 ||
      resource_info.SizeInBytes == std::numeric_limits<uint64_t>::max()) {
    // Invalid resource
    return nullptr;
  }

  ComPtr<ID3D12Resource> committed_resource;
  // D3D12 creates an implicit heap that contains the resource allocation when
  // calling CreateCommittedResource.
  // TODO: Store a heap object for every allocated ResourceAllocation that will
  // be managed by residency management.
  d3d12_device->CreateCommittedResource(
      &heap_properties, D3D12_HEAP_FLAG_NONE, &resource_descriptor,
      initial_usage, nullptr, IID_PPV_ARGS(&committed_resource));

  return committed_resource;
}

// TODO
inline void CopyBufferRegionUtil(ComPtr<ID3D12GraphicsCommandList> commandList,
                             ComPtr<ID3D12Resource> srcResource,
                             ComPtr<ID3D12Resource> destResource,
                             UINT64 resourceSize,
                             D3D12_RESOURCE_STATES state,
                             bool needBarrierEnd = true) {
  D3D12_RESOURCE_BARRIER resourceBarrier;
  if (state == D3D12_RESOURCE_STATE_COPY_DEST) {
    resourceBarrier.Transition.pResource = destResource.Get();
  } else if (state == D3D12_RESOURCE_STATE_COPY_SOURCE) {
    resourceBarrier.Transition.pResource = srcResource.Get();
  } else {
    LOG(ERROR) << "Unsupported D3D12_RESOURCE_STATES.";
    assert(0);
  }
  resourceBarrier.Transition.StateBefore =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  resourceBarrier.Transition.StateAfter = state;
  resourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  resourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  resourceBarrier.Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  commandList->ResourceBarrier(1, &resourceBarrier);
  commandList->CopyBufferRegion(destResource.Get(), 0, srcResource.Get(), 0,
                                resourceSize);
  if (needBarrierEnd) {
    resourceBarrier.Transition.StateBefore = state;
    resourceBarrier.Transition.StateAfter =
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    commandList->ResourceBarrier(1, &resourceBarrier);
  }
}

inline D3D12_HEAP_PROPERTIES CreateHeapProperties(
    D3D12_HEAP_TYPE type = D3D12_HEAP_TYPE_DEFAULT) {
  return {type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1,
          1};
};

inline D3D12_RESOURCE_DESC CreateResourceDesc(
    UINT64 width,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
  return {D3D12_RESOURCE_DIMENSION_BUFFER,
          0,
          width,
          1,
          1,
          1,
          DXGI_FORMAT_UNKNOWN,
          {1, 0},
          D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
          flags};
};

template <typename T>
T RoundUpToMultiple(T value, T multiple) {
  static_assert(std::is_integral_v<T>);

  T remainder = value % multiple;
  if (remainder != 0) {
    value += multiple - remainder;
  }

  return value;
}

// Represent the information of the graph's edges.
struct EdgeInfoBase {
  virtual ~EdgeInfoBase() = default;
  DML_TENSOR_DESC outputTensorDESC = {};
  std::string name = "";
  bool isInputEdge = false;
};

// Only represent the information of the input edges.
struct InputEdgeInfo final : public EdgeInfoBase {
  InputEdgeInfo();
  ~InputEdgeInfo() override;
  // Indicate the index of the graph's input.
  size_t inputIndex = 0;
  size_t byteLength = 0;
  ComPtr<ID3D12Resource> resource;
  // Indicate if the input is from constant buffer which need to be
  // uploaded in the stage of initialization.
  bool isConstantInput = false;
};

// Represent the information of the intermediate edges and output edges.
struct EdgeInfo final : public EdgeInfoBase {
  ~EdgeInfo() override = default;
  // Indicate the index of the intermediate node from which this edge was
  // produced.
  uint32_t nodeIndex = 0;
  // Indicate the index of the intermediate node' output from which this edge
  // was produced.
  uint32_t outputNodeIndex = 0;
};

inline void CloseExecuteResetWait(
    ComPtr<ID3D12GraphicsCommandList> commandList,
    ComPtr<ID3D12CommandQueue> commandQueue,
    ComPtr<ID3D12CommandAllocator> commandAllocator,
    ComPtr<ID3D12Device> D3D12Device) {
  WEBNN_CHECK(commandList->Close());
  ID3D12CommandList* commandLists[] = {commandList.Get()};
  commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
  WEBNN_CHECK(
      commandQueue.Get()->GetDevice(IID_PPV_ARGS(D3D12Device.GetAddressOf())));
  ComPtr<ID3D12Fence> fence;
  WEBNN_CHECK(D3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                       IID_PPV_ARGS(fence.GetAddressOf())));
  WEBNN_CHECK(commandQueue.Get()->Signal(fence.Get(), 1));
  WEBNN_CHECK(fence->SetEventOnCompletion(1, nullptr));
  WEBNN_CHECK(commandAllocator->Reset());
  WEBNN_CHECK(commandList->Reset(commandAllocator.Get(), nullptr));
}

template <typename T>
void ComputeImplicitPaddingForAutoPad(AutoPad auto_pad,
                                      T dilation,
                                      T inputSize,
                                      T filterSize,
                                      T stride,
                                      T& paddingBegin,
                                      T& paddingEnd) {
  T outSize = (inputSize + stride - 1) / stride;
  T dilatedFilter = (filterSize - 1) * dilation + 1;
  T neededInput = (outSize - 1) * stride + dilatedFilter;
  T totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
  switch (auto_pad) {
    case AutoPad::kSameUpper:
      paddingBegin = totalPadding / 2;
      paddingEnd = (totalPadding + 1) / 2;
      break;
    case AutoPad::kSameLower:
      paddingBegin = (totalPadding + 1) / 2;
      paddingEnd = totalPadding / 2;
      break;
    default:
      assert(0);
  }
}

template <typename S, typename T>
std::vector<T> ComputeImplicitPaddingForAutoPad(const S* options,
                                                std::vector<T> inputSize,
                                                std::vector<T> filterSize) {
  std::vector<T> padding(4);
  ComputeImplicitPaddingForAutoPad<T>(
      options->auto_pad, options->dilations[0], inputSize[0], filterSize[0],
      options->strides[0], padding[0], padding[1]);
  ComputeImplicitPaddingForAutoPad<T>(
      options->auto_pad, options->dilations[1], inputSize[1], filterSize[1],
      options->strides[1], padding[2], padding[3]);
  return padding;
}

template <typename T>
void ComputeImplicitPaddingForConvTranspose2dAutoPad(AutoPad auto_pad,
                                                     T dilation,
                                                     T inputSize,
                                                     T filterSize,
                                                     T stride,
                                                     T outputPadding,
                                                     T& paddingBegin,
                                                     T& paddingEnd) {
  T outSize = inputSize * stride;
  T totalPadding = stride * (inputSize - 1) + outputPadding +
                   ((filterSize - 1) * dilation + 1) - outSize;
  switch (auto_pad) {
    case AutoPad::kSameUpper:
      paddingBegin = totalPadding / 2;
      paddingEnd = totalPadding - totalPadding / 2;
      break;
    case AutoPad::kSameLower:
      paddingBegin = totalPadding - totalPadding / 2;
      paddingEnd = totalPadding / 2;
      break;
    default:
      assert(0);
  }
}

// template <typename T>
// std::vector<T> ComputeImplicitPaddingForConvTranspose2dAutoPad(
//     const ConvTranspose2dOptions* options,
//     std::vector<T> inputSize,
//     std::vector<T> filterSize) {
//     std::vector<T> padding(4);
//     utils::ComputeImplicitPaddingForConvTranspose2dAutoPad<T>(
//         options->auto_pad, options->dilations[0], inputSize[0],
//         filterSize[0], options->strides[0], options->outputPadding[0],
//         padding[0], padding[1]);
//     utils::ComputeImplicitPaddingForConvTranspose2dAutoPad<T>(
//         options->auto_pad, options->dilations[1], inputSize[1],
//         filterSize[1], options->strides[1], options->outputPadding[1],
//         padding[2], padding[3]);
//     return padding;
// }

}  // namespace content::webnn

#endif  // WEBNN_NATIVE_DML_UTILS_H_
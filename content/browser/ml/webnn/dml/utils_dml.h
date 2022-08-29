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

template <typename T>
T RoundUpToMultiple(T value, T multiple) {
  static_assert(std::is_integral_v<T>);

  T remainder = value % multiple;
  if (remainder != 0) {
    value += multiple - remainder;
  }

  return value;
}

// An adapter called the "Microsoft Basic Render Driver" is always present. This
// adapter is a render-only device that has no display outputs.
HRESULT IsWarpAdapter(IDXGIAdapter1* pAdapter, bool* isWarpAdapter) {
  DXGI_ADAPTER_DESC1 pDesc;
  WEBNN_CHECK(pAdapter->GetDesc1(&pDesc));
  // see here for documentation on filtering WARP adapter:
  // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
  auto isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
  auto isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
  auto isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
  *isWarpAdapter = isSoftwareAdapter ||
                   (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);
  return S_OK;
}

void CloseExecuteResetWait(ComPtr<ID3D12GraphicsCommandList> commandList,
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
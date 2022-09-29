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
#include "components/ml/mojom/webnn_graph.mojom.h"

#define WEBNN_CHECK(hr)                   \
  if (((HRESULT)(hr)) < 0) {              \
    LOG(ERROR) << "Failed to do " << #hr; \
    assert(0);                            \
  }

namespace content::webnn {

using namespace Microsoft::WRL;
using ml::webnn::mojom::AutoPad;

// Round up to alignment
inline size_t Align(size_t value, UINT alignment) {
  size_t remainder = value % alignment;
  if (remainder != 0) {
    value += alignment - remainder;
  }

  return value;
}

template <typename T>
std::pair<size_t, T> Align(T& memory_info_map, UINT alignment) {
  T aligned_memory_info_map;
  size_t aligned_offset = 0;
  for (auto& [key, memory_info] : memory_info_map) {
    uint64_t aligned_byte_length = Align(memory_info->byte_length, alignment);
    auto aligned_memory_info = ml::webnn::mojom::MemoryInfo::New();
    aligned_memory_info->byte_offset = aligned_offset;
    aligned_memory_info->byte_length = aligned_byte_length;
    aligned_memory_info_map[key] = std::move(aligned_memory_info);
    aligned_offset += aligned_byte_length;
  }

  return std::make_pair(aligned_offset, std::move(aligned_memory_info_map));
}

inline std::vector<UINT> ConvertDimensions(
    const std::vector<int32_t>& dimensions) {
  std::vector<UINT> convertedDimensions;
  for (auto dim : dimensions) {
    if (dim < 0) {
      LOG(ERROR) << "DML doesn't support the negative dimension value";
      assert(0);
    }
    convertedDimensions.push_back(dim);
  }
  return convertedDimensions;
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
std::vector<UINT> ImplicitPadding(const T* options,
                                  const std::vector<UINT>& inputDims,
                                  const std::vector<UINT>& filterDims) {
  return ComputeImplicitPaddingForAutoPad<T, UINT>(
      options, {inputDims[2], inputDims[3]},
      {filterDims[filterDims.size() - 2], filterDims[filterDims.size() - 1]});
}

template <typename T>
std::vector<UINT> ExplicitPadding(const T* options) {
  UINT paddingTop = static_cast<UINT>(options->padding[0]);
  UINT paddingBottom = static_cast<UINT>(options->padding[1]);
  UINT paddingLeft = static_cast<UINT>(options->padding[2]);
  UINT paddingRight = static_cast<UINT>(options->padding[3]);

  return {paddingTop, paddingBottom, paddingLeft, paddingRight};
}

}  // namespace content::webnn

#endif  // WEBNN_NATIVE_DML_UTILS_H_
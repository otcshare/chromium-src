// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_GRAPH_TENSOR_DESC_H_
#define CONTENT_BROWSER_ML_WEBNN_GRAPH_TENSOR_DESC_H_

#include <vector>

#include <wrl.h>
#include "DirectML.h"
#include "third_party/abseil-cpp/absl/types/optional.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;

class TensorDesc final {
 public:
  TensorDesc();
  TensorDesc(DML_TENSOR_DATA_TYPE data_type, std::vector<UINT> dimensions);
  TensorDesc(DML_TENSOR_DATA_TYPE data_type,
             DML_TENSOR_FLAGS flags,
             std::vector<UINT> dimensions);
  TensorDesc(DML_TENSOR_DATA_TYPE data_type,
             DML_TENSOR_FLAGS flags,
             std::vector<UINT> dimensions,
             absl::optional<std::vector<UINT>> strides);
  TensorDesc(TensorDesc&& other);
  TensorDesc& operator=(TensorDesc&& other);
  ~TensorDesc();

  DML_TENSOR_DESC* Get();
  DML_TENSOR_DATA_TYPE GetDataType() const;
  DML_TENSOR_FLAGS GetFlags() const;
  std::vector<UINT>& GetDimensions();
  absl::optional<std::vector<UINT>>& GetStrides();
  UINT64 GetTotalTensorSizeInBytes();

 private:
  void Initialize(DML_TENSOR_DATA_TYPE data_type,
                  DML_TENSOR_FLAGS flags,
                  std::vector<UINT> dimensions,
                  absl::optional<std::vector<UINT>> strides);
  // DML_BUFFER_TENSOR_DESC only has a pointer of dimensions and strides, the
  // data hold in dimensions_ and strides_.
  std::vector<UINT> dimensions_;
  absl::optional<std::vector<UINT>> strides_;
  // Describes a tensor that will be stored in a Direct3D 12 buffer resource.
  DML_BUFFER_TENSOR_DESC buffer_desc_ = {};
  DML_TENSOR_DESC tensor_desc_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_GRAPH_TENSOR_DESC_H_

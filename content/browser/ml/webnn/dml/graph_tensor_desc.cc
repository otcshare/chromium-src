// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/graph_tensor_desc.h"

#include "base/check_op.h"
#include "base/numerics/checked_math.h"

namespace content::webnn {

namespace {

size_t GetBytesOfDataType(DML_TENSOR_DATA_TYPE data_type) {
  switch (data_type) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      return sizeof(float);
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      return sizeof(uint16_t);
    case DML_TENSOR_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case DML_TENSOR_DATA_TYPE_UINT32:
      return sizeof(uint32_t);
    default:
      return 0;
  }
}

absl::optional<UINT64> CalculateElementsNumber(
    const std::vector<UINT>& dimensions) {
  base::CheckedNumeric<UINT64> checked_elements_number = 1;
  for (auto& d : dimensions) {
    checked_elements_number *= d;
  }
  if (!checked_elements_number.IsValid()) {
    return absl::nullopt;
  }
  return checked_elements_number.ValueOrDie();
}

absl::optional<UINT64> CalculateElementsNumber(
    const std::vector<UINT>& dimensions,
    const absl::optional<std::vector<UINT>>& strides) {
  if (!strides) {
    return CalculateElementsNumber(dimensions);
  }

  base::CheckedNumeric<UINT64> checked_elements_number = 1;
  for (uint32_t i = 0; i < dimensions.size(); ++i) {
    // The specific dimension from broadcasting shouldn't increase the count of
    // elements.
    auto d = dimensions[i];
    if (strides.value()[i] == 0) {
      d = 1;
    }
    checked_elements_number *= d;
  }
  if (!checked_elements_number.IsValid()) {
    return absl::nullopt;
  }
  return checked_elements_number.ValueOrDie();
}

absl::optional<UINT64> TotalTensorSizeInBytes(
    DML_TENSOR_DATA_TYPE data_type,
    const std::vector<UINT>& dimensions,
    const absl::optional<std::vector<UINT>>& strides) {
  absl::optional<size_t> elements_num =
      CalculateElementsNumber(dimensions, strides);
  if (!elements_num) {
    return absl::nullopt;
  }

  base::CheckedNumeric<UINT64> checked_total_size_in_bytes =
      elements_num.value() * GetBytesOfDataType(data_type);
  if (!checked_total_size_in_bytes.IsValid()) {
    return absl::nullopt;
  }
  return checked_total_size_in_bytes.ValueOrDie();
}

}  // namespace

TensorDesc::TensorDesc() = default;

TensorDesc::TensorDesc(DML_TENSOR_DATA_TYPE data_type,
                       std::vector<UINT> dimensions) {
  Initialize(data_type, DML_TENSOR_FLAG_NONE, std::move(dimensions),
             absl::nullopt);
}

TensorDesc::TensorDesc(DML_TENSOR_DATA_TYPE data_type,
                       DML_TENSOR_FLAGS flags,
                       std::vector<UINT> dimensions) {
  Initialize(data_type, flags, std::move(dimensions), absl::nullopt);
}

TensorDesc::TensorDesc(DML_TENSOR_DATA_TYPE data_type,
                       DML_TENSOR_FLAGS flags,
                       std::vector<UINT> dimensions,
                       absl::optional<std::vector<UINT>> strides) {
  Initialize(data_type, flags, std::move(dimensions), std::move(strides));
}

void TensorDesc::Initialize(DML_TENSOR_DATA_TYPE data_type,
                            DML_TENSOR_FLAGS flags,
                            std::vector<UINT> dimensions,
                            absl::optional<std::vector<UINT>> strides) {
  DCHECK(!dimensions.empty() &&
         dimensions.size() < DML_TENSOR_DIMENSION_COUNT_MAX);
  DCHECK(!strides || dimensions.size() == strides->size());
  dimensions_ = std::move(dimensions);
  strides_ = std::move(strides);

  buffer_desc_.DimensionCount = dimensions_.size();
  buffer_desc_.Sizes = dimensions_.data();
  buffer_desc_.Strides = strides_ ? strides_.value().data() : nullptr;
  buffer_desc_.TotalTensorSizeInBytes =
      TotalTensorSizeInBytes(data_type, dimensions_, strides_).value();
  buffer_desc_.GuaranteedBaseOffsetAlignment = 0;
  buffer_desc_.DataType = data_type;
  buffer_desc_.Flags = flags;
}

TensorDesc::TensorDesc(TensorDesc&& other) = default;
TensorDesc& TensorDesc::operator=(TensorDesc&& other) = default;

TensorDesc::~TensorDesc() = default;

DML_TENSOR_DESC* TensorDesc::Get() {
  if (buffer_desc_.DataType == DML_TENSOR_DATA_TYPE_UNKNOWN) {
    return nullptr;
  }
  tensor_desc_ = DML_TENSOR_DESC{DML_TENSOR_TYPE_BUFFER, &buffer_desc_};
  return &tensor_desc_;
}

DML_TENSOR_DATA_TYPE TensorDesc::GetDataType() const {
  return buffer_desc_.DataType;
}

DML_TENSOR_FLAGS TensorDesc::GetFlags() const {
  return buffer_desc_.Flags;
}

std::vector<UINT>& TensorDesc::GetDimensions() {
  return dimensions_;
}

absl::optional<std::vector<UINT>>& TensorDesc::GetStrides() {
  return strides_;
}

UINT64 TensorDesc::GetTotalTensorSizeInBytes() {
  return buffer_desc_.TotalTensorSizeInBytes;
}

}  // namespace content::webnn

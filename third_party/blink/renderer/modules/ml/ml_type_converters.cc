// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_type_converters.h"

#include <iostream>

#include "third_party/blink/public/mojom/ml/ml.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_union_stringsequence_stringunsignedlongrecord.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_load_model_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/core/typed_arrays/array_buffer/array_buffer_contents.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_typed_array.h"
#include "third_party/blink/renderer/modules/modules_export.h"

namespace mojo {

using ml::mojom::blink::Float32List;
using ml::mojom::blink::Float64List;
using ml::mojom::blink::Int16List;
using ml::mojom::blink::Int32List;
using ml::mojom::blink::Int64List;
using ml::mojom::blink::Int8List;
using ml::mojom::blink::LoadModelIOSpec;
using ml::mojom::blink::LoadModelOptions;
using ml::mojom::blink::LoadModelOptionsPtr;
using ml::mojom::blink::Tensor;
using ml::mojom::blink::TensorPtr;
using ml::mojom::blink::Uint16List;
using ml::mojom::blink::Uint32List;
using ml::mojom::blink::Uint64List;
using ml::mojom::blink::Uint8List;
using ml::mojom::blink::ValueList;

// static
LoadModelOptionsPtr
TypeConverter<LoadModelOptionsPtr, blink::MLLoadModelOptions*>::Convert(
    const blink::MLLoadModelOptions* input) {
  if (!input) {
    return nullptr;
  }
  auto output = LoadModelOptions::New();
  // Converts `MLLoadModelOptions::inputs`.
  if (input->inputs() != nullptr) {
    if (input->inputs()->IsStringSequence()) {
      auto inputs = LoadModelIOSpec::New();
      inputs->set_array_of_strings(input->inputs()->GetAsStringSequence());
      output->inputs = std::move(inputs);
    } else if (input->inputs()->IsStringUnsignedLongRecord()) {
      HashMap<String, uint64_t> map_mojo;
      for (const auto& name_indice :
           input->inputs()->GetAsStringUnsignedLongRecord()) {
        map_mojo.insert(name_indice.first, name_indice.second);
      }
      auto inputs = LoadModelIOSpec::New();
      inputs->set_names_to_indices(std::move(map_mojo));
      output->inputs = std::move(inputs);
    } else {
      NOTREACHED();
    }
  }
  // Converts `MLLoadModelOptions::outputs`.
  if (input->outputs() != nullptr) {
    if (input->outputs()->IsStringSequence()) {
      auto outputs = LoadModelIOSpec::New();
      outputs->set_array_of_strings(input->outputs()->GetAsStringSequence());
      output->outputs = std::move(outputs);
    } else if (input->outputs()->IsStringUnsignedLongRecord()) {
      HashMap<String, uint64_t> map_mojo;
      for (const auto& name_indice :
           input->outputs()->GetAsStringUnsignedLongRecord()) {
        map_mojo.insert(name_indice.first, name_indice.second);
      }
      auto outputs = LoadModelIOSpec::New();
      outputs->set_names_to_indices(std::move(map_mojo));
      output->outputs = std::move(outputs);
    } else {
      NOTREACHED();
    }
  }
  return output;
}

// static
TensorPtr TypeConverter<TensorPtr, blink::MLTensor*>::Convert(
    const blink::MLTensor* input) {
  if (!input) {
    return nullptr;
  }
  auto output = Tensor::New();
  // Convert data.
#define ML_CONVERT_BLINK_ARRAY_TO_MOJO(mojo_lower, mojo_upper, blink_upper) \
  {                                                                         \
    auto* mojo_lower##array_blink =                                         \
        static_cast<blink::DOM##blink_upper##Array*>(input->data().Get());  \
    auto mojo_lower##array_mojo = mojo_upper##List::New();                  \
    for (size_t i = 0; i < mojo_lower##array_blink->length(); i++) {        \
      mojo_lower##array_mojo->value.push_back(                              \
          mojo_lower##array_blink->Item(i));                                \
    }                                                                       \
    auto value_list = ValueList::New();                                     \
    value_list->set_##mojo_lower##_list(std::move(mojo_lower##array_mojo)); \
    output->data = std::move(value_list);                                   \
    break;                                                                  \
  }

  switch (input->data()->GetType()) {
    case blink::DOMArrayBufferView::kTypeInt8:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(int8, Int8, Int8);
    case blink::DOMArrayBufferView::kTypeUint8:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(uint8, Uint8, Uint8);
    case blink::DOMArrayBufferView::kTypeInt16:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(int16, Int16, Int16);
    case blink::DOMArrayBufferView::kTypeUint16:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(uint16, Uint16, Uint16);
    case blink::DOMArrayBufferView::kTypeInt32:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(int32, Int32, Int32);
    case blink::DOMArrayBufferView::kTypeUint32:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(uint32, Uint32, Uint32);
    case blink::DOMArrayBufferView::kTypeBigInt64:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(int64, Int64, BigInt64);
    case blink::DOMArrayBufferView::kTypeBigUint64:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(uint64, Uint64, BigUint64);
    case blink::DOMArrayBufferView::kTypeFloat32:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(float32, Float32, Float32);
    case blink::DOMArrayBufferView::kTypeFloat64:
      ML_CONVERT_BLINK_ARRAY_TO_MOJO(float64, Float64, Float64);
    default:
      // For non-supported types, returns nullptr.
      return nullptr;
  }
#undef ML_CONVERT_BLINK_ARRAY_TO_MOJO

  // Convert dimensions.
  auto dimensions = Uint32List::New();
  for (const auto v : input->dimensions()) {
    dimensions->value.push_back(v);
  }
  output->dimensions = std::move(dimensions);

  return output;
}

// Converters from Mojo to IDL.
// static
blink::MLTensor*
TypeConverter<blink::MLTensor*, ml::mojom::blink::TensorPtr>::Convert(
    const ml::mojom::blink::TensorPtr& input) {
  if (!input)
    return nullptr;

  auto* output = blink::MLTensor::Create();

  // Convert data.
#define ML_CONVERT_MOJO_ARRAY_TO_BLINK(mojo_lower, blink_upper)               \
  const auto& value = input->data->get_##mojo_lower##_list()->value;          \
  auto* typed_array = blink::DOM##blink_upper##Array::Create(value.size());   \
  for (wtf_size_t i = 0; i < value.size(); i++) {                             \
    typed_array->Data()[i] = value[i];                                        \
  }                                                                           \
  blink::NotShared<blink::DOMArrayBufferView> array_buffer_view(typed_array); \
  output->setData(std::move(array_buffer_view));

  if (input->data) {
    if (input->data->is_float32_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(float32, Float32);
    } else if (input->data->is_float64_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(float64, Float64);
    } else if (input->data->is_int8_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(int8, Int8);
    } else if (input->data->is_uint8_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(uint8, Uint8);
    } else if (input->data->is_int16_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(int16, Int16);
    } else if (input->data->is_uint16_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(uint16, Uint16);
    } else if (input->data->is_int32_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(int32, Int32);
    } else if (input->data->is_uint32_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(uint32, Uint32);
    } else if (input->data->is_int64_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(int64, BigInt64);
    } else if (input->data->is_uint64_list()) {
      ML_CONVERT_MOJO_ARRAY_TO_BLINK(uint64, BigUint64);
    } else {
      // For non-supported types, returns nullptr.
      return nullptr;
    }
  } else {
    return nullptr;
  }
  // Convert dimensions.
  if (input->dimensions) {
    output->setDimensions(input->dimensions->value);
  } else {
    return nullptr;
  }

  return output;
}

}  // namespace mojo

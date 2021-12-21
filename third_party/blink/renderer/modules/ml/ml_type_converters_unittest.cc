// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <limits>
#include <memory>
#include <utility>

#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/blink/public/mojom/ml/ml.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_union_stringsequence_stringunsignedlongrecord.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_load_model_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_typedefs.h"
#include "third_party/blink/renderer/modules/ml/ml_type_converters.h"

namespace blink {

using ml::mojom::blink::Float32List;
using ml::mojom::blink::Float64List;
using ml::mojom::blink::Int16List;
using ml::mojom::blink::Int32List;
using ml::mojom::blink::Int64List;
using ml::mojom::blink::Int8List;
using ml::mojom::blink::LoadModelOptionsPtr;
using ml::mojom::blink::TensorPtr;
using ml::mojom::blink::Uint16List;
using ml::mojom::blink::Uint32List;
using ml::mojom::blink::Uint64List;
using ml::mojom::blink::Uint8List;
using ml::mojom::blink::ValueList;

TEST(MLTypeConvertersTest, IdlLoadModelOptionsToMojoStringSequenceValue) {
  blink::MLLoadModelOptions idl_options;
  Vector<String> value_input{"input_1", "input_2"};
  Vector<String> value_output{"output_1", "output_2", "output_3"};

  blink::V8MLLoadModelIOSpec inputs(std::move(value_input));
  blink::V8MLLoadModelIOSpec outputs(std::move(value_output));

  idl_options.setInputs(&inputs);
  idl_options.setOutputs(&outputs);

  auto mojo_options = mojo::ConvertTo<LoadModelOptionsPtr>(&idl_options);

  ASSERT_FALSE(mojo_options.is_null());

  ASSERT_TRUE(mojo_options->inputs->is_array_of_strings());
  ASSERT_EQ(mojo_options->inputs->get_array_of_strings().size(), 2u);
  EXPECT_EQ(mojo_options->inputs->get_array_of_strings()[0], "input_1");
  EXPECT_EQ(mojo_options->inputs->get_array_of_strings()[1], "input_2");

  ASSERT_TRUE(mojo_options->outputs->is_array_of_strings());
  ASSERT_EQ(mojo_options->outputs->get_array_of_strings().size(), 3u);
  EXPECT_EQ(mojo_options->outputs->get_array_of_strings()[0], "output_1");
  EXPECT_EQ(mojo_options->outputs->get_array_of_strings()[1], "output_2");
  EXPECT_EQ(mojo_options->outputs->get_array_of_strings()[2], "output_3");
}

TEST(MLTypeConvertersTest,
     IdlLoadModelOptionsToMojoStingUnsignedLongRecordValue) {
  blink::MLLoadModelOptions idl_options;
  Vector<std::pair<String, uint32_t>> value_input{{"key_1", 1}, {"key_2", 2}};
  Vector<std::pair<String, uint32_t>> value_output{
      {"key_3", 3}, {"key_4", 4}, {"key_5", 5}};

  blink::V8MLLoadModelIOSpec inputs(std::move(value_input));
  blink::V8MLLoadModelIOSpec outputs(std::move(value_output));

  idl_options.setInputs(&inputs);
  idl_options.setOutputs(&outputs);

  auto mojo_options = mojo::ConvertTo<LoadModelOptionsPtr>(&idl_options);

  ASSERT_FALSE(mojo_options.is_null());

  ASSERT_TRUE(mojo_options->inputs->is_names_to_indices());
  ASSERT_EQ(mojo_options->inputs->get_names_to_indices().size(), 2u);
  ASSERT_TRUE(mojo_options->inputs->get_names_to_indices().Contains("key_1"));
  EXPECT_EQ(mojo_options->inputs->get_names_to_indices().at("key_1"), 1u);
  ASSERT_TRUE(mojo_options->inputs->get_names_to_indices().Contains("key_2"));
  EXPECT_EQ(mojo_options->inputs->get_names_to_indices().at("key_2"), 2u);

  ASSERT_TRUE(mojo_options->outputs->is_names_to_indices());
  ASSERT_EQ(mojo_options->outputs->get_names_to_indices().size(), 3u);
  ASSERT_TRUE(mojo_options->outputs->get_names_to_indices().Contains("key_3"));
  EXPECT_EQ(mojo_options->outputs->get_names_to_indices().at("key_3"), 3u);
  ASSERT_TRUE(mojo_options->outputs->get_names_to_indices().Contains("key_4"));
  EXPECT_EQ(mojo_options->outputs->get_names_to_indices().at("key_4"), 4u);
  ASSERT_TRUE(mojo_options->outputs->get_names_to_indices().Contains("key_5"));
  EXPECT_EQ(mojo_options->outputs->get_names_to_indices().at("key_5"), 5u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoUnsupportedType) {
  // Tests int8 conversion.
  blink::MLTensor idl_tensor_int8clamped;
  auto* int8clamped_array = blink::DOMUint8ClampedArray::Create(3);
  int8clamped_array->Data()[0] = 10;
  int8clamped_array->Data()[1] = 20;
  int8clamped_array->Data()[2] = 30;

  blink::NotShared<blink::DOMArrayBufferView> int8clamped_buffer_view(
      int8clamped_array);
  idl_tensor_int8clamped.setData(std::move(int8clamped_buffer_view));

  idl_tensor_int8clamped.setDimensions({1, 1, 3});

  auto mojo_tensor_int8clamped =
      mojo::ConvertTo<TensorPtr>(&idl_tensor_int8clamped);

  ASSERT_TRUE(mojo_tensor_int8clamped.is_null());
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoInt8) {
  // Tests int8 conversion.
  blink::MLTensor idl_tensor_int8;
  auto* int8_array = blink::DOMInt8Array::Create(3);
  int8_array->Data()[0] = 10;
  int8_array->Data()[1] = 20;
  int8_array->Data()[2] = 30;

  blink::NotShared<blink::DOMArrayBufferView> int8_buffer_view(int8_array);
  idl_tensor_int8.setData(std::move(int8_buffer_view));

  idl_tensor_int8.setDimensions({1, 1, 3});

  auto mojo_tensor_int8 = mojo::ConvertTo<TensorPtr>(&idl_tensor_int8);

  ASSERT_FALSE(mojo_tensor_int8.is_null());

  ASSERT_FALSE(mojo_tensor_int8->data.is_null());
  ASSERT_TRUE(mojo_tensor_int8->data->is_int8_list());
  ASSERT_EQ(mojo_tensor_int8->data->get_int8_list()->value.size(), 3u);
  EXPECT_EQ(mojo_tensor_int8->data->get_int8_list()->value[0],
            static_cast<int8_t>(10));
  EXPECT_EQ(mojo_tensor_int8->data->get_int8_list()->value[1],
            static_cast<int8_t>(20));
  EXPECT_EQ(mojo_tensor_int8->data->get_int8_list()->value[2],
            static_cast<int8_t>(30));

  ASSERT_FALSE(mojo_tensor_int8->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_int8->dimensions->value.size(), 3u);
  ASSERT_EQ(mojo_tensor_int8->dimensions->value[0], 1u);
  ASSERT_EQ(mojo_tensor_int8->dimensions->value[1], 1u);
  ASSERT_EQ(mojo_tensor_int8->dimensions->value[2], 3u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoUint8) {
  // Tests uint8 conversion.
  blink::MLTensor idl_tensor_uint8;
  auto* uint8_array = blink::DOMUint8Array::Create(2);
  uint8_array->Data()[0] = 10;
  uint8_array->Data()[1] = 255;

  blink::NotShared<blink::DOMArrayBufferView> uint8_buffer_view(uint8_array);
  idl_tensor_uint8.setData(std::move(uint8_buffer_view));

  idl_tensor_uint8.setDimensions({2, 1});

  auto mojo_tensor_uint8 = mojo::ConvertTo<TensorPtr>(&idl_tensor_uint8);

  ASSERT_FALSE(mojo_tensor_uint8.is_null());

  ASSERT_FALSE(mojo_tensor_uint8->data.is_null());
  ASSERT_TRUE(mojo_tensor_uint8->data->is_uint8_list());
  ASSERT_EQ(mojo_tensor_uint8->data->get_uint8_list()->value.size(), 2u);
  EXPECT_EQ(mojo_tensor_uint8->data->get_uint8_list()->value[0],
            static_cast<uint8_t>(10));
  EXPECT_EQ(mojo_tensor_uint8->data->get_uint8_list()->value[1],
            static_cast<uint8_t>(255));

  ASSERT_FALSE(mojo_tensor_uint8->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_uint8->dimensions->value.size(), 2u);
  ASSERT_EQ(mojo_tensor_uint8->dimensions->value[0], 2u);
  ASSERT_EQ(mojo_tensor_uint8->dimensions->value[1], 1u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoInt16) {
  // Tests int16 conversion.
  blink::MLTensor idl_tensor_int16;
  auto* int16_array = blink::DOMInt16Array::Create(2);
  int16_array->Data()[0] = 10;
  int16_array->Data()[1] = -1000;

  blink::NotShared<blink::DOMArrayBufferView> int16_buffer_view(int16_array);
  idl_tensor_int16.setData(std::move(int16_buffer_view));

  idl_tensor_int16.setDimensions({2, 1, 1});

  auto mojo_tensor_int16 = mojo::ConvertTo<TensorPtr>(&idl_tensor_int16);

  ASSERT_FALSE(mojo_tensor_int16.is_null());

  ASSERT_FALSE(mojo_tensor_int16->data.is_null());
  ASSERT_TRUE(mojo_tensor_int16->data->is_int16_list());
  ASSERT_EQ(mojo_tensor_int16->data->get_int16_list()->value.size(), 2u);
  EXPECT_EQ(mojo_tensor_int16->data->get_int16_list()->value[0],
            static_cast<int16_t>(10));
  EXPECT_EQ(mojo_tensor_int16->data->get_int16_list()->value[1],
            static_cast<int16_t>(-1000));

  ASSERT_FALSE(mojo_tensor_int16->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_int16->dimensions->value.size(), 3u);
  ASSERT_EQ(mojo_tensor_int16->dimensions->value[0], 2u);
  ASSERT_EQ(mojo_tensor_int16->dimensions->value[1], 1u);
  ASSERT_EQ(mojo_tensor_int16->dimensions->value[2], 1u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoUint16) {
  // Tests uint16 conversion.
  blink::MLTensor idl_tensor_uint16;
  auto* uint16_array = blink::DOMUint16Array::Create(2);
  uint16_array->Data()[0] = 10;
  uint16_array->Data()[1] = 1000;

  blink::NotShared<blink::DOMArrayBufferView> uint16_buffer_view(uint16_array);
  idl_tensor_uint16.setData(std::move(uint16_buffer_view));

  idl_tensor_uint16.setDimensions({2, 1, 1});

  auto mojo_tensor_uint16 = mojo::ConvertTo<TensorPtr>(&idl_tensor_uint16);

  ASSERT_FALSE(mojo_tensor_uint16.is_null());

  ASSERT_FALSE(mojo_tensor_uint16->data.is_null());
  ASSERT_TRUE(mojo_tensor_uint16->data->is_uint16_list());
  ASSERT_EQ(mojo_tensor_uint16->data->get_uint16_list()->value.size(), 2u);
  EXPECT_EQ(mojo_tensor_uint16->data->get_uint16_list()->value[0],
            static_cast<uint16_t>(10));
  EXPECT_EQ(mojo_tensor_uint16->data->get_uint16_list()->value[1],
            static_cast<uint16_t>(1000));

  ASSERT_FALSE(mojo_tensor_uint16->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_uint16->dimensions->value.size(), 3u);
  ASSERT_EQ(mojo_tensor_uint16->dimensions->value[0], 2u);
  ASSERT_EQ(mojo_tensor_uint16->dimensions->value[1], 1u);
  ASSERT_EQ(mojo_tensor_uint16->dimensions->value[2], 1u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoInt32) {
  // Tests int32 conversion.
  blink::MLTensor idl_tensor_int32;
  auto* int32_array = blink::DOMInt32Array::Create(3);
  int32_array->Data()[0] = 10;
  int32_array->Data()[1] = std::numeric_limits<int32_t>::max();
  int32_array->Data()[2] = std::numeric_limits<int32_t>::min();

  blink::NotShared<blink::DOMArrayBufferView> int32_buffer_view(int32_array);
  idl_tensor_int32.setData(std::move(int32_buffer_view));

  idl_tensor_int32.setDimensions({1, 3, 1});

  auto mojo_tensor_int32 = mojo::ConvertTo<TensorPtr>(&idl_tensor_int32);

  ASSERT_FALSE(mojo_tensor_int32.is_null());

  ASSERT_FALSE(mojo_tensor_int32->data.is_null());
  ASSERT_TRUE(mojo_tensor_int32->data->is_int32_list());
  ASSERT_EQ(mojo_tensor_int32->data->get_int32_list()->value.size(), 3u);
  EXPECT_EQ(mojo_tensor_int32->data->get_int32_list()->value[0], 10);
  EXPECT_EQ(mojo_tensor_int32->data->get_int32_list()->value[1],
            std::numeric_limits<int32_t>::max());
  EXPECT_EQ(mojo_tensor_int32->data->get_int32_list()->value[2],
            std::numeric_limits<int32_t>::min());

  ASSERT_FALSE(mojo_tensor_int32->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_int32->dimensions->value.size(), 3u);
  ASSERT_EQ(mojo_tensor_int32->dimensions->value[0], 1u);
  ASSERT_EQ(mojo_tensor_int32->dimensions->value[1], 3u);
  ASSERT_EQ(mojo_tensor_int32->dimensions->value[2], 1u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoUint32) {
  // Tests uint32 conversion.
  blink::MLTensor idl_tensor_uint32;
  auto* uint32_array = blink::DOMUint32Array::Create(4);
  uint32_array->Data()[0] = 10;
  uint32_array->Data()[1] = 0;
  uint32_array->Data()[2] = std::numeric_limits<uint32_t>::max();
  uint32_array->Data()[3] = std::numeric_limits<uint32_t>::min();

  blink::NotShared<blink::DOMArrayBufferView> uint32_buffer_view(uint32_array);
  idl_tensor_uint32.setData(std::move(uint32_buffer_view));

  idl_tensor_uint32.setDimensions({2, 1, 1, 2});

  auto mojo_tensor_uint32 = mojo::ConvertTo<TensorPtr>(&idl_tensor_uint32);

  ASSERT_FALSE(mojo_tensor_uint32.is_null());

  ASSERT_FALSE(mojo_tensor_uint32->data.is_null());
  ASSERT_TRUE(mojo_tensor_uint32->data->is_uint32_list());
  ASSERT_EQ(mojo_tensor_uint32->data->get_uint32_list()->value.size(), 4u);
  EXPECT_EQ(mojo_tensor_uint32->data->get_uint32_list()->value[0], 10u);
  EXPECT_EQ(mojo_tensor_uint32->data->get_uint32_list()->value[1], 0u);
  EXPECT_EQ(mojo_tensor_uint32->data->get_uint32_list()->value[2],
            std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(mojo_tensor_uint32->data->get_uint32_list()->value[3],
            std::numeric_limits<uint32_t>::min());

  ASSERT_FALSE(mojo_tensor_uint32->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_uint32->dimensions->value.size(), 4u);
  ASSERT_EQ(mojo_tensor_uint32->dimensions->value[0], 2u);
  ASSERT_EQ(mojo_tensor_uint32->dimensions->value[1], 1u);
  ASSERT_EQ(mojo_tensor_uint32->dimensions->value[2], 1u);
  ASSERT_EQ(mojo_tensor_uint32->dimensions->value[3], 2u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoInt64) {
  // Tests int64 conversion.
  blink::MLTensor idl_tensor_int64;
  auto* int64_array = blink::DOMBigInt64Array::Create(4);
  int64_array->Data()[0] = 10;
  int64_array->Data()[1] = 0;
  int64_array->Data()[2] = std::numeric_limits<int64_t>::max();
  int64_array->Data()[3] = std::numeric_limits<int64_t>::min();

  blink::NotShared<blink::DOMArrayBufferView> int64_buffer_view(int64_array);
  idl_tensor_int64.setData(std::move(int64_buffer_view));

  idl_tensor_int64.setDimensions({2, 1, 1, 2});

  auto mojo_tensor_int64 = mojo::ConvertTo<TensorPtr>(&idl_tensor_int64);

  ASSERT_FALSE(mojo_tensor_int64.is_null());

  ASSERT_FALSE(mojo_tensor_int64->data.is_null());
  ASSERT_TRUE(mojo_tensor_int64->data->is_int64_list());
  ASSERT_EQ(mojo_tensor_int64->data->get_int64_list()->value.size(), 4u);
  EXPECT_EQ(mojo_tensor_int64->data->get_int64_list()->value[0], 10u);
  EXPECT_EQ(mojo_tensor_int64->data->get_int64_list()->value[1], 0u);
  EXPECT_EQ(mojo_tensor_int64->data->get_int64_list()->value[2],
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ(mojo_tensor_int64->data->get_int64_list()->value[3],
            std::numeric_limits<int64_t>::min());

  ASSERT_FALSE(mojo_tensor_int64->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_int64->dimensions->value.size(), 4u);
  ASSERT_EQ(mojo_tensor_int64->dimensions->value[0], 2u);
  ASSERT_EQ(mojo_tensor_int64->dimensions->value[1], 1u);
  ASSERT_EQ(mojo_tensor_int64->dimensions->value[2], 1u);
  ASSERT_EQ(mojo_tensor_int64->dimensions->value[3], 2u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoUint64) {
  // Tests uint64 conversion.
  blink::MLTensor idl_tensor_uint64;
  auto* uint64_array = blink::DOMBigUint64Array::Create(5);
  uint64_array->Data()[0] = 10;
  uint64_array->Data()[1] = 0;
  uint64_array->Data()[2] = std::numeric_limits<uint64_t>::max();
  uint64_array->Data()[3] = std::numeric_limits<uint64_t>::min();
  uint64_array->Data()[4] = 12345;

  blink::NotShared<blink::DOMArrayBufferView> uint64_buffer_view(uint64_array);
  idl_tensor_uint64.setData(std::move(uint64_buffer_view));

  idl_tensor_uint64.setDimensions({1, 5, 1});

  auto mojo_tensor_uint64 = mojo::ConvertTo<TensorPtr>(&idl_tensor_uint64);

  ASSERT_FALSE(mojo_tensor_uint64.is_null());

  ASSERT_FALSE(mojo_tensor_uint64->data.is_null());
  ASSERT_TRUE(mojo_tensor_uint64->data->is_uint64_list());
  ASSERT_EQ(mojo_tensor_uint64->data->get_uint64_list()->value.size(), 5u);
  EXPECT_EQ(mojo_tensor_uint64->data->get_uint64_list()->value[0], 10ull);
  EXPECT_EQ(mojo_tensor_uint64->data->get_uint64_list()->value[1], 0ull);
  EXPECT_EQ(mojo_tensor_uint64->data->get_uint64_list()->value[2],
            std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(mojo_tensor_uint64->data->get_uint64_list()->value[3],
            std::numeric_limits<uint64_t>::min());
  EXPECT_EQ(mojo_tensor_uint64->data->get_uint64_list()->value[4], 12345ull);

  ASSERT_FALSE(mojo_tensor_uint64->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_uint64->dimensions->value.size(), 3u);
  ASSERT_EQ(mojo_tensor_uint64->dimensions->value[0], 1u);
  ASSERT_EQ(mojo_tensor_uint64->dimensions->value[1], 5u);
  ASSERT_EQ(mojo_tensor_uint64->dimensions->value[2], 1u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoFloat32) {
  // Tests float32 conversion.
  blink::MLTensor idl_tensor_float32;
  auto* float32_array = blink::DOMFloat32Array::Create(5);
  float32_array->Data()[0] = 10;
  float32_array->Data()[1] = 0;
  float32_array->Data()[2] = std::numeric_limits<float>::max();
  float32_array->Data()[3] = std::numeric_limits<float>::min();
  float32_array->Data()[4] = 12345;

  blink::NotShared<blink::DOMArrayBufferView> float32_buffer_view(
      float32_array);
  idl_tensor_float32.setData(std::move(float32_buffer_view));

  idl_tensor_float32.setDimensions({1, 5, 1});

  auto mojo_tensor_float32 = mojo::ConvertTo<TensorPtr>(&idl_tensor_float32);

  ASSERT_FALSE(mojo_tensor_float32.is_null());

  ASSERT_FALSE(mojo_tensor_float32->data.is_null());
  ASSERT_TRUE(mojo_tensor_float32->data->is_float32_list());
  ASSERT_EQ(mojo_tensor_float32->data->get_float32_list()->value.size(), 5u);
  EXPECT_NEAR(mojo_tensor_float32->data->get_float32_list()->value[0], 10.,
              1e-5);
  EXPECT_NEAR(mojo_tensor_float32->data->get_float32_list()->value[1], 0.,
              1e-5);
  EXPECT_NEAR(mojo_tensor_float32->data->get_float32_list()->value[2],
              std::numeric_limits<float>::max(), 1e-5);
  EXPECT_NEAR(mojo_tensor_float32->data->get_float32_list()->value[3],
              std::numeric_limits<float>::min(), 1e-5);
  EXPECT_NEAR(mojo_tensor_float32->data->get_float32_list()->value[4], 12345.,
              1e-5);

  ASSERT_FALSE(mojo_tensor_float32->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_float32->dimensions->value.size(), 3u);
  ASSERT_EQ(mojo_tensor_float32->dimensions->value[0], 1u);
  ASSERT_EQ(mojo_tensor_float32->dimensions->value[1], 5u);
  ASSERT_EQ(mojo_tensor_float32->dimensions->value[2], 1u);
}

TEST(MLTypeConvertersTest, IdlMLTensorToMojoFloat64) {
  // Tests float64 conversion.
  blink::MLTensor idl_tensor_float64;
  auto* float64_array = blink::DOMFloat64Array::Create(5);
  float64_array->Data()[0] = std::numeric_limits<double>::max();
  float64_array->Data()[1] = 0;
  float64_array->Data()[2] = -2.3;
  float64_array->Data()[3] = std::numeric_limits<double>::min();
  float64_array->Data()[4] = 12.345;

  blink::NotShared<blink::DOMArrayBufferView> float64_buffer_view(
      float64_array);
  idl_tensor_float64.setData(std::move(float64_buffer_view));

  idl_tensor_float64.setDimensions({1, 1, 5});

  auto mojo_tensor_float64 = mojo::ConvertTo<TensorPtr>(&idl_tensor_float64);

  ASSERT_FALSE(mojo_tensor_float64.is_null());

  ASSERT_FALSE(mojo_tensor_float64->data.is_null());
  ASSERT_TRUE(mojo_tensor_float64->data->is_float64_list());
  ASSERT_EQ(mojo_tensor_float64->data->get_float64_list()->value.size(), 5u);
  EXPECT_EQ(mojo_tensor_float64->data->get_float64_list()->value[0],
            std::numeric_limits<double>::max());
  EXPECT_NEAR(mojo_tensor_float64->data->get_float64_list()->value[1], 0, 1e-5);
  EXPECT_NEAR(mojo_tensor_float64->data->get_float64_list()->value[2], -2.3,
              1e-5);
  EXPECT_NEAR(mojo_tensor_float64->data->get_float64_list()->value[3],
              std::numeric_limits<double>::min(), 1e-5);
  EXPECT_NEAR(mojo_tensor_float64->data->get_float64_list()->value[4], 12.345,
              1e-5);

  ASSERT_FALSE(mojo_tensor_float64->dimensions.is_null());
  ASSERT_EQ(mojo_tensor_float64->dimensions->value.size(), 3u);
  ASSERT_EQ(mojo_tensor_float64->dimensions->value[0], 1u);
  ASSERT_EQ(mojo_tensor_float64->dimensions->value[1], 1u);
  ASSERT_EQ(mojo_tensor_float64->dimensions->value[2], 5u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlEmptyInput) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_EQ(idl_tensor, nullptr);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlEmptyData) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);
  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(1);

  ASSERT_EQ(idl_tensor, nullptr);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlEmptyDimension) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_int8_list(Int8List::New());
  value_list->get_int8_list()->value.push_back(10);
  mojo_tensor->data = std::move(value_list);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_EQ(idl_tensor, nullptr);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlInt8) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_int8_list(Int8List::New());
  value_list->get_int8_list()->value.push_back(10);
  value_list->get_int8_list()->value.push_back(-20);
  value_list->get_int8_list()->value.push_back(-100);
  value_list->get_int8_list()->value.push_back(110);
  value_list->get_int8_list()->value.push_back(30);
  value_list->get_int8_list()->value.push_back(40);
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(2);
  mojo_tensor->dimensions->value.push_back(3);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeInt8);
  const auto* idl_typed_array =
      static_cast<blink::DOMInt8Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 6u);
  EXPECT_EQ(idl_typed_array->Item(0), 10);
  EXPECT_EQ(idl_typed_array->Item(1), -20);
  EXPECT_EQ(idl_typed_array->Item(2), -100);
  EXPECT_EQ(idl_typed_array->Item(3), 110);
  EXPECT_EQ(idl_typed_array->Item(4), 30);
  EXPECT_EQ(idl_typed_array->Item(5), 40);

  ASSERT_EQ(idl_tensor->dimensions().size(), 3u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 2u);
  EXPECT_EQ(idl_tensor->dimensions()[2], 3u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlUint8) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_uint8_list(Uint8List::New());
  value_list->get_uint8_list()->value.push_back(10);
  value_list->get_uint8_list()->value.push_back(
      std::numeric_limits<uint8_t>::min());
  value_list->get_uint8_list()->value.push_back(
      std::numeric_limits<uint8_t>::max());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(3);
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(1);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeUint8);
  const auto* idl_typed_array =
      static_cast<blink::DOMUint8Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 3u);
  EXPECT_EQ(idl_typed_array->Item(0), 10);
  EXPECT_EQ(idl_typed_array->Item(1), std::numeric_limits<uint8_t>::min());
  EXPECT_EQ(idl_typed_array->Item(2), std::numeric_limits<uint8_t>::max());

  ASSERT_EQ(idl_tensor->dimensions().size(), 3u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 3u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[2], 1u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlInt16) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_int16_list(Int16List::New());
  value_list->get_int16_list()->value.push_back(-543);
  value_list->get_int16_list()->value.push_back(100);
  value_list->get_int16_list()->value.push_back(
      std::numeric_limits<int16_t>::min());
  value_list->get_int16_list()->value.push_back(30);
  value_list->get_int16_list()->value.push_back(
      std::numeric_limits<int16_t>::max());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(5);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeInt16);
  const auto* idl_typed_array =
      static_cast<blink::DOMInt16Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 5u);
  EXPECT_EQ(idl_typed_array->Item(0), -543);
  EXPECT_EQ(idl_typed_array->Item(1), 100);
  EXPECT_EQ(idl_typed_array->Item(2), std::numeric_limits<int16_t>::min());
  EXPECT_EQ(idl_typed_array->Item(3), 30);
  EXPECT_EQ(idl_typed_array->Item(4), std::numeric_limits<int16_t>::max());

  ASSERT_EQ(idl_tensor->dimensions().size(), 3u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[2], 5u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlUint16) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_uint16_list(Uint16List::New());
  value_list->get_uint16_list()->value.push_back(423);
  value_list->get_uint16_list()->value.push_back(
      std::numeric_limits<uint16_t>::min());
  value_list->get_uint16_list()->value.push_back(
      std::numeric_limits<uint16_t>::max());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(3);
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(1);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeUint16);
  const auto* idl_typed_array =
      static_cast<blink::DOMUint16Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 3u);
  EXPECT_EQ(idl_typed_array->Item(0), 423u);
  EXPECT_EQ(idl_typed_array->Item(1), std::numeric_limits<uint16_t>::min());
  EXPECT_EQ(idl_typed_array->Item(2), std::numeric_limits<uint16_t>::max());

  ASSERT_EQ(idl_tensor->dimensions().size(), 3u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 3u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[2], 1u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlInt32) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_int32_list(Int32List::New());
  value_list->get_int32_list()->value.push_back(1234);
  value_list->get_int32_list()->value.push_back(-3431);
  value_list->get_int32_list()->value.push_back(
      std::numeric_limits<int32_t>::min());
  value_list->get_int32_list()->value.push_back(
      std::numeric_limits<int32_t>::max());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(4);
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(1);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeInt32);
  const auto* idl_typed_array =
      static_cast<blink::DOMInt32Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 4u);
  EXPECT_EQ(idl_typed_array->Item(0), 1234);
  EXPECT_EQ(idl_typed_array->Item(1), -3431);
  EXPECT_EQ(idl_typed_array->Item(2), std::numeric_limits<int32_t>::min());
  EXPECT_EQ(idl_typed_array->Item(3), std::numeric_limits<int32_t>::max());

  ASSERT_EQ(idl_tensor->dimensions().size(), 3u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 4u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[2], 1u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlUint32) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_uint32_list(Uint32List::New());
  value_list->get_uint32_list()->value.push_back(1);
  value_list->get_uint32_list()->value.push_back(54321);
  value_list->get_uint32_list()->value.push_back(
      std::numeric_limits<uint32_t>::min());
  value_list->get_uint32_list()->value.push_back(
      std::numeric_limits<uint32_t>::max());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(2);
  mojo_tensor->dimensions->value.push_back(2);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeUint32);
  const auto* idl_typed_array =
      static_cast<blink::DOMUint32Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 4u);
  EXPECT_EQ(idl_typed_array->Item(0), 1u);
  EXPECT_EQ(idl_typed_array->Item(1), 54321u);
  EXPECT_EQ(idl_typed_array->Item(2), std::numeric_limits<uint32_t>::min());
  EXPECT_EQ(idl_typed_array->Item(3), std::numeric_limits<uint32_t>::max());

  ASSERT_EQ(idl_tensor->dimensions().size(), 3u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 2u);
  EXPECT_EQ(idl_tensor->dimensions()[2], 2u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlInt64) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_int64_list(Int64List::New());
  value_list->get_int64_list()->value.push_back(
      std::numeric_limits<int64_t>::min());
  value_list->get_int64_list()->value.push_back(-12345);
  value_list->get_int64_list()->value.push_back(
      std::numeric_limits<int64_t>::max());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(1);
  mojo_tensor->dimensions->value.push_back(3);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeBigInt64);
  const auto* idl_typed_array =
      static_cast<blink::DOMBigInt64Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 3u);
  EXPECT_EQ(idl_typed_array->Item(0), std::numeric_limits<int64_t>::min());
  EXPECT_EQ(idl_typed_array->Item(1), -12345);
  EXPECT_EQ(idl_typed_array->Item(2), std::numeric_limits<int64_t>::max());

  ASSERT_EQ(idl_tensor->dimensions().size(), 3u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 1u);
  EXPECT_EQ(idl_tensor->dimensions()[2], 3u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlUint64) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_uint64_list(Uint64List::New());
  value_list->get_uint64_list()->value.push_back(
      std::numeric_limits<uint64_t>::min());
  value_list->get_uint64_list()->value.push_back(100);
  value_list->get_uint64_list()->value.push_back(110);
  value_list->get_uint64_list()->value.push_back(30);
  value_list->get_uint64_list()->value.push_back(
      std::numeric_limits<uint64_t>::max());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(5);
  mojo_tensor->dimensions->value.push_back(1);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeBigUint64);
  const auto* idl_typed_array =
      static_cast<blink::DOMBigUint64Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 5u);
  EXPECT_EQ(idl_typed_array->Item(0), std::numeric_limits<uint64_t>::min());
  EXPECT_EQ(idl_typed_array->Item(1), 100u);
  EXPECT_EQ(idl_typed_array->Item(2), 110u);
  EXPECT_EQ(idl_typed_array->Item(3), 30u);
  EXPECT_EQ(idl_typed_array->Item(4), std::numeric_limits<uint64_t>::max());

  ASSERT_EQ(idl_tensor->dimensions().size(), 2u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 5u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 1u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlFloat32) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_float32_list(Float32List::New());
  value_list->get_float32_list()->value.push_back(
      std::numeric_limits<float>::max());
  value_list->get_float32_list()->value.push_back(100);
  value_list->get_float32_list()->value.push_back(
      std::numeric_limits<float>::min());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(3);
  mojo_tensor->dimensions->value.push_back(1);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeFloat32);
  const auto* idl_typed_array =
      static_cast<blink::DOMFloat32Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 3u);
  EXPECT_NEAR(idl_typed_array->Item(0), std::numeric_limits<float>::max(),
              1e-5);
  EXPECT_NEAR(idl_typed_array->Item(1), 100., 1e-5);
  EXPECT_NEAR(idl_typed_array->Item(2), std::numeric_limits<float>::min(),
              1e-5);

  ASSERT_EQ(idl_tensor->dimensions().size(), 2u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 3u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 1u);
}

TEST(MLTypeConvertersTest, MojoMLTensorToIdlFloat64) {
  auto mojo_tensor = ml::mojom::blink::Tensor::New();
  auto value_list = ValueList::New();
  value_list->set_float64_list(Float64List::New());
  value_list->get_float64_list()->value.push_back(10);
  value_list->get_float64_list()->value.push_back(
      std::numeric_limits<double>::max());
  value_list->get_float64_list()->value.push_back(110);
  value_list->get_float64_list()->value.push_back(
      std::numeric_limits<double>::min());
  mojo_tensor->data = std::move(value_list);

  mojo_tensor->dimensions = Uint32List::New();
  mojo_tensor->dimensions->value.push_back(2);
  mojo_tensor->dimensions->value.push_back(2);
  mojo_tensor->dimensions->value.push_back(1);

  auto* idl_tensor = mojo::ConvertTo<blink::MLTensor*>(mojo_tensor);

  ASSERT_NE(idl_tensor, nullptr);

  ASSERT_EQ(idl_tensor->data()->GetType(),
            blink::DOMArrayBufferView::kTypeFloat64);
  const auto* idl_typed_array =
      static_cast<blink::DOMFloat64Array*>(idl_tensor->data().Get());
  ASSERT_EQ(idl_typed_array->length(), 4u);
  EXPECT_NEAR(idl_typed_array->Item(0), 10., 1e-5);
  EXPECT_NEAR(idl_typed_array->Item(1), std::numeric_limits<double>::max(),
              1e-5);
  EXPECT_NEAR(idl_typed_array->Item(2), 110., 1e-5);
  EXPECT_NEAR(idl_typed_array->Item(3), std::numeric_limits<double>::min(),
              1e-5);

  ASSERT_EQ(idl_tensor->dimensions().size(), 3u);
  EXPECT_EQ(idl_tensor->dimensions()[0], 2u);
  EXPECT_EQ(idl_tensor->dimensions()[1], 2u);
  EXPECT_EQ(idl_tensor->dimensions()[2], 1u);
}

}  // namespace blink

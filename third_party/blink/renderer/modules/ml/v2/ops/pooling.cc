// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/pooling.h"

#include <memory>

#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

namespace blink {

Pooling::Pooling(Operand* input,
                 WTF::Vector<int32_t> window_dimensions,
                 WTF::Vector<int32_t> padding,
                 WTF::Vector<int32_t> strides,
                 WTF::Vector<int32_t> dilations,
                 String layout,
                 PoolingType type)
    : Output({input}),
      window_dimensions_(std::move(window_dimensions)),
      padding_(std::move(padding)),
      strides_(std::move(strides)),
      dilations_(std::move(dilations)),
      layout_(std::move(layout)),
      type_(type) {}

void Pooling::AddLayer(NNModel* model, uint32_t& index) {
  Vector<uint32_t> input_indexes;
  // Add input index to input_indexes.
  for (auto& input : Output::Inputs()) {
    input_indexes.push_back(input->Index());
  }

  // Add padding opeand and set the value.
  for (auto padding : padding_) {
    uint32_t padding_index = index++;
    model->AddScalarOperand(padding_index, padding);
    input_indexes.push_back(padding_index);
  }

  // Add strides Operand and set the value.
  for (auto stride : strides_) {
    uint32_t stride_index = index++;
    model->AddScalarOperand(stride_index, stride);
    input_indexes.push_back(stride_index);
  }

  // Add filters Operand and set the value.
  for (auto filter : window_dimensions_) {
    uint32_t filter_index = index++;
    model->AddScalarOperand(filter_index, filter);
    input_indexes.push_back(filter_index);
  }

  // Add fused code operand and set the value.
  uint32_t fuse_index = index++;
  model->AddScalarOperand(fuse_index, 0);
  input_indexes.push_back(fuse_index);

  // Add layout operand and set the value.
  uint32_t layout_index = index++;
  model->AddScalarOperand(layout_index, layout_ == "nchw" ? 1 : 0);
  input_indexes.push_back(layout_index);

  // Add conv output operand.
  uint32_t output_index = index++;
  Operand::SetIndex(output_index);
  model->AddUnspecifiedOperand();

  int32_t operation_type = -1;
  switch (type_) {
    case kPoolingTypeAverage:
      operation_type = NeuralNetworkContext::kAveragePool2D;
      break;
    case kPoolingTypeMax:
      operation_type = NeuralNetworkContext::kMaxPool2D;
      break;
    default:
      LOG(ERROR) << "The operation isn't supported";
      NOTREACHED();
  }

  model->AddOperation(operation_type, input_indexes, {output_index});
}

}  // namespace blink

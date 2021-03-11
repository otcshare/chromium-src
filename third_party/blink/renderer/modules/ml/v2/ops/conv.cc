// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/conv.h"

#include <memory>

#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

namespace blink {

namespace {

uint32_t GetOutputChannel(Member<Operand>& operand, const String& layout) {
  Vector<uint32_t> dimensions = operand->GetDimensions();
  if (dimensions.IsEmpty())
    return 0;

  if (layout == "nchw") {
    return dimensions[0];
  } else {
    return dimensions[3];
  }
}

}  // namespace

Conv::Conv(Operand* input,
           Operand* filter,
           WTF::Vector<int32_t> padding,
           WTF::Vector<int32_t> strides,
           WTF::Vector<int32_t> dilations,
           int32_t groups,
           String layout)
    : Output({input, filter}),
      padding_(std::move(padding)),
      strides_(std::move(strides)),
      dilations_(std::move(dilations)),
      groups_(groups),
      layout_(std::move(layout)) {}

void Conv::AddLayer(NNModel* model, uint32_t& index) {
  Vector<uint32_t> input_indexes;
  // Add input and filter index to input_indexes.
  for (auto& input : Output::Inputs()) {
    input_indexes.push_back(input->Index());
  }

  // Add a empty bias operand that is used in Android NN API.
  uint32_t bias_index = index++;
  model->AddBiasOperand(bias_index,
                        GetOutputChannel(Output::Inputs()[1], layout_));
  input_indexes.push_back(bias_index);

  // Add padding opeand and set the value.
  for (auto padding : padding_) {
    uint32_t padding_index = index++;
    model->AddScalarOperand(padding_index, padding);
    input_indexes.push_back(padding_index);
  }

  // Add strides / dilations Operand and set the value.
  bool atrous = product(dilations_) != 1 ? true : false;
  bool depthwise = groups_ != 1 ? true : false;
  // The new design still need to be align with Android NN API in order to work
  // on Android Platform, but we can only use explicit padding.
  int32_t operation_type = depthwise ? NeuralNetworkContext::kDepthwiseConv2D
                                     : NeuralNetworkContext::kConv2D;
  if (atrous) {
    for (auto dilation : dilations_) {
      uint32_t dilation_index = index++;
      model->AddScalarOperand(dilation_index, dilation);
      input_indexes.push_back(dilation_index);
    }
    operation_type = depthwise ? NeuralNetworkContext::kAtrousDepthwiseConv2D
                               : NeuralNetworkContext::kAtrousConv2D;
  } else {
    for (auto stride : strides_) {
      uint32_t stride_index = index++;
      model->AddScalarOperand(stride_index, stride);
      input_indexes.push_back(stride_index);
    }
  }

  if (depthwise) {
    uint32_t depthwise_index = index++;
    model->AddScalarOperand(depthwise_index, groups_);
    input_indexes.push_back(depthwise_index);
  }

  // Add fused code operand and set the value.
  uint32_t fuse_index = index++;
  model->AddScalarOperand(fuse_index, 0);
  input_indexes.push_back(fuse_index);

  // Add layout operand and set the value.
  uint32_t layout_index = index++;
  model->AddScalarOperand(layout_index, layout_ == "nchw" ? true : false);
  input_indexes.push_back(layout_index);

  // Add conv output operand.
  uint32_t output_index = index++;
  Operand::SetIndex(output_index);
  model->AddUnspecifiedOperand();

  model->AddOperation(operation_type, input_indexes, {output_index});
}

}  // namespace blink

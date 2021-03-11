// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/reshape.h"

#include <memory>

#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

namespace blink {

Reshape::Reshape(Operand* input, WTF::Vector<int32_t> new_shape)
    : Output({input}), new_shape_(new_shape) {}

void Reshape::AddLayer(NNModel* model, uint32_t& index) {
  Vector<uint32_t> input_indexes;
  // Add input index to input_indexes.
  for (auto& input : Output::Inputs()) {
    input_indexes.push_back(input->Index());
  }

  // Add new shape operand and set the value.
  uint32_t new_shape_index = index++;
  // The new shape is 1-D tensor.
  Vector<uint32_t> new_shape_dims(1, new_shape_.size());
  model->AddTensorOperand(new_shape_index, new_shape_dims, new_shape_);
  input_indexes.push_back(new_shape_index);

  // Add Reshape output operand.
  uint32_t output_index = index++;
  Operand::SetIndex(output_index);
  model->AddUnspecifiedOperand();

  model->AddOperation(NeuralNetworkContext::kReshape, input_indexes,
                      {output_index});
}

}  // namespace blink

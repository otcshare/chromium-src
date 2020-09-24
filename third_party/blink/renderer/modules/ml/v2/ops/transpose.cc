// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/transpose.h"

#include <memory>

#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

namespace blink {

Transpose::Transpose(Operand* input, WTF::Vector<int32_t> permutation)
    : Output({input}), permutation_(permutation) {}

void Transpose::AddLayer(NNModel* model, uint32_t& index) {
  Vector<uint32_t> input_indexes;
  // Add input index to input_indexes.
  for (auto& input : Output::Inputs()) {
    input_indexes.push_back(input->Index());
  }

  // Add permutation operand and set the value.
  if (!permutation_.IsEmpty()) {
    uint32_t permutation_index = index++;
    // The new shape is 1-D tensor.
    Vector<uint32_t> permutation_dims(1, permutation_.size());
    model->AddTensorOperand(permutation_index, permutation_dims, permutation_);
    input_indexes.push_back(permutation_index);
  }

  // Add Reshape output operand.
  uint32_t output_index = index++;
  Operand::SetIndex(output_index);
  model->AddUnspecifiedOperand();

  model->AddOperation(NeuralNetworkContext::kTranspose, input_indexes,
                      {output_index});
}

}  // namespace blink

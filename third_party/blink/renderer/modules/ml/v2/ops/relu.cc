// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/relu.h"

#include <memory>

#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

namespace blink {

Relu::Relu(Operand* input) : Output({input}) {}

void Relu::AddLayer(NNModel* model, uint32_t& index) {
  Vector<uint32_t> input_indexes;
  // Add input index to input_indexes.
  for (auto& input : Output::Inputs()) {
    input_indexes.push_back(input->Index());
  }

  // Add Softmax output operand.
  uint32_t output_index = index++;
  Operand::SetIndex(output_index);
  model->AddUnspecifiedOperand();

  model->AddOperation(NeuralNetworkContext::kRelu, input_indexes,
                      {output_index});
}

}  // namespace blink

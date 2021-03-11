// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/matmul.h"

#include <memory>

#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

namespace blink {

MatMul::MatMul(Operand* a, Operand* b) : Output({a, b}) {}

void MatMul::AddLayer(NNModel* model, uint32_t& index) {
  Vector<uint32_t> input_indexes;
  // Add input index to input_indexes.
  for (auto& input : Output::Inputs()) {
    input_indexes.push_back(input->Index());
  }

  // We can't get the bias size.
  uint32_t bias_index = index++;
  model->AddUnspecifiedOperand();
  input_indexes.push_back(bias_index);

  // Add fused code operand and set the value.
  uint32_t fuse_index = index++;
  model->AddScalarOperand(fuse_index, 0);
  input_indexes.push_back(fuse_index);

  // There are no MatMul defined in Android NN API, We use kFullyConnected
  // instead of MatMul.
  uint32_t matmul_index = index++;
  model->AddScalarOperand(matmul_index, 0);
  input_indexes.push_back(matmul_index);

  // Add MatMul output operand.
  uint32_t output_index = index++;
  Operand::SetIndex(output_index);
  model->AddUnspecifiedOperand();

  model->AddOperation(NeuralNetworkContext::kFullyConnected, input_indexes,
                      {output_index});
}

}  // namespace blink

// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/binary.h"

#include "third_party/blink/renderer/modules/ml/neural_network_context.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

namespace blink {

Binary::Binary(BinaryType type, Operand* primary, Operand* secondary)
    : Output({primary, secondary}), type_(type) {}

void Binary::AddLayer(NNModel* model, uint32_t& index) {
  // Add FuseCode Operand defined in Android NN API.
  uint32_t fuse_index = index++;
  model->AddScalarOperand(fuse_index, 0);

  // Add element-wise binary output operand.
  uint32_t output_index = index++;
  Operand::SetIndex(output_index);
  model->AddUnspecifiedOperand();
  // addOperation
  int32_t operation_type = -1;
  switch (type_) {
    case kBinaryTypeAdd:
      operation_type = NeuralNetworkContext::kAdd;
      break;
    case kBinaryTypeMul:
      operation_type = NeuralNetworkContext::kMul;
      break;
    default:
      LOG(ERROR) << "The operation isn't supported";
      NOTREACHED();
  }

  // Add a.index and b.index to inputs.
  HeapVector<Member<Operand>>& operand_inputs = Output::Inputs();
  Vector<uint32_t> inputs = {operand_inputs[0]->Index(),
                             operand_inputs[1]->Index(), fuse_index};
  model->AddOperation(operation_type, inputs, {output_index});
}

}  // namespace blink

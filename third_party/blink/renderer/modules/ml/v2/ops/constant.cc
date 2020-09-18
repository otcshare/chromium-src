// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/constant.h"

namespace blink {

Constant::Constant(const OperandDescriptor* descriptor,
                   DOMArrayBufferView* data)
    : Operand(),
      descriptor_(const_cast<OperandDescriptor*>(descriptor)),
      data_(data) {}

void Constant::AddLayer(NNModel* model, uint32_t& index) {
  // Add operand for constant.
  uint32_t constant_index = index++;
  Operand::SetIndex(constant_index);
  model->AddOperand(descriptor_);

  // setOperandValue
  model->SetOperandValue(constant_index, data_);
}

Vector<uint32_t> Constant::GetDimensions() {
  return descriptor_->dimensions();
}

void Constant::Trace(Visitor* visitor) const {
  visitor->Trace(descriptor_);
  visitor->Trace(data_);
  Operand::Trace(visitor);
}

}  // namespace blink

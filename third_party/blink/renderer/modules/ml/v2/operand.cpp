// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/operand.h"

// #include "third_party/blink/renderer/platform/bindings/enumeration_base.h"
#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

namespace blink {

Operand* Operand::FirstInput() const {
  return nullptr;
}

Operand* Operand::NextInput() {
  return nullptr;
}

void Operand::AddLayer(NNModel* model, uint32_t& index) {
  LOG(INFO) << "Operand.";
}

Vector<uint32_t> Operand::GetDimensions() {
  return Vector<uint32_t>();
}

void Operand::SetIndex(uint32_t index) {
  index_ = index;
}

uint32_t Operand::Index() {
  return index_;
}

void Operand::SetTraversal(bool traversal) {
  traversalled_ = traversal;
}

bool Operand::Traversal() {
  return traversalled_;
}

void Operand::Trace(Visitor* visitor) const {
  ScriptWrappable::Trace(visitor);
}

}  // namespace blink

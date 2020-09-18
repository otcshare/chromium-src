// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_CONSTANT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_CONSTANT_H_

#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_model.h"
#include "third_party/blink/renderer/modules/ml/v2/operand.h"
#include "third_party/blink/renderer/modules/ml/v2/operand_descriptor.h"

namespace blink {

class Constant final : public Operand {
 public:
  Constant(const OperandDescriptor*, DOMArrayBufferView*);
  ~Constant() override = default;

  void AddLayer(NNModel* model, uint32_t& index) override;
  Vector<uint32_t> GetDimensions() override;

  // Interface required by garbage collection.
  void Trace(Visitor*) const override;

 private:
  String name_;
  Member<OperandDescriptor> descriptor_;
  Member<DOMArrayBufferView> data_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_ADD_H_

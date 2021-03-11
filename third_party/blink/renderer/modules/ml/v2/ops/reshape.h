// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_RESHAPE_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_RESHAPE_H_

#include "third_party/blink/renderer/modules/ml/v2/nn_model.h"
#include "third_party/blink/renderer/modules/ml/v2/operand.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/output.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

namespace blink {

class Reshape final : public Output {
 public:
  Reshape(Operand*, WTF::Vector<int32_t>);
  ~Reshape() override = default;

  void AddLayer(NNModel* model, uint32_t& index) override;

 private:
  Vector<int32_t> new_shape_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_RESHAPE_H_

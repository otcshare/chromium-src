// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_POOLING_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_POOLING_H_

#include "third_party/blink/renderer/modules/ml/v2/nn_model.h"
#include "third_party/blink/renderer/modules/ml/v2/operand.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/output.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

namespace blink {

enum PoolingType {
  kPoolingTypeAverage = 0,
  kPoolingTypeL2,
  kPoolingTypeMax,
};

class Pooling final : public Output {
 public:
  Pooling(Operand*,
          WTF::Vector<int32_t>,
          WTF::Vector<int32_t>,
          WTF::Vector<int32_t>,
          WTF::Vector<int32_t>,
          String,
          PoolingType);
  ~Pooling() override = default;

  void AddLayer(NNModel* model, uint32_t& index) override;

 private:
  Vector<int32_t> window_dimensions_;
  Vector<int32_t> padding_;
  Vector<int32_t> strides_;
  Vector<int32_t> dilations_;
  String layout_;
  PoolingType type_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_POOLING_H_

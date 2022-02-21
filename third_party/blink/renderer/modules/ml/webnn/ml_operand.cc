// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink {

MLOperand::MLOperand(MLContext* context, WNNOperand operand)
    : WebnnObject<WNNOperand>(context, operand) {
}

void MLOperand::Trace(Visitor* visitor) const {
  WebnnObject<WNNOperand>::Trace(visitor);
}

}  // namespace blink
// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERAND_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERAND_H_

#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"

namespace blink {

class MLOperand : public WebnnObject<WNNOperand> {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit MLOperand(MLContext* context,
                     WNNOperand operand);

  MLOperand(const MLOperand&) = delete;
  MLOperand& operator=(const MLOperand&) = delete;

  void Trace(Visitor* visitor) const override;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERAND_H_

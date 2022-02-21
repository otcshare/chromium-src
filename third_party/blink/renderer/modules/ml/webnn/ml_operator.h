// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_OPERATOR_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_OPERATOR_H_

#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"

namespace blink {

class MLOperator : public WebnnObject<WNNFusionOperator> {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit MLOperator(MLContext* context,
                     WNNFusionOperator op);

  MLOperator(const MLOperator&) = delete;
  MLOperator& operator=(const MLOperator&) = delete;

  void Trace(Visitor* visitor) const override;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_OPERATOR_H_

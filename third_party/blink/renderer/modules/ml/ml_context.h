// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_

#include <webnn/webnn.h>

#include "third_party/blink/renderer/modules/ml/webnn/webnn_context.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/graphics/gpu/webnn_control_client_holder.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

class ML;
class WebnnContext;
class ExecutionContext;

class MLContext final : public WebnnContext {
  DEFINE_WRAPPERTYPEINFO();

 public:
  MLContext(ExecutionContext* execution_context,
            scoped_refptr<WebnnControlClientHolder> webnn_control_client,
            WNNContext webnn_context,
            const AtomicString model_format,
            ML* ml);

  MLContext(const MLContext&) = delete;
  MLContext& operator=(const MLContext&) = delete;

  ~MLContext() override;

  AtomicString modelFormat() const;

  ML* GetML();

  void Trace(Visitor* visitor) const override;

 private:
  AtomicString model_format_;

  Member<ML> ml_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_

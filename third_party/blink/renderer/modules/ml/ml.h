// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_

#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/frame/navigator.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_device.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

class MLContextOptions;
class ScriptPromise;
class ScriptState;
class WebnnInstance;

class ML final : public ScriptWrappable,
                 public ExecutionContextLifecycleObserver {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit ML(ExecutionContext* execution_context);

  ML(const ML&) = delete;
  ML& operator=(const ML&) = delete;

  void CreateModelLoader(ScriptState* script_state,
                         ExceptionState& exception_state);

  // ScriptWrappable overrides
  void Trace(blink::Visitor*) const override;

  // ExecutionContextLifecycleObserver overrides
  void ContextDestroyed() override;

  // IDL interface:
  MLContext* createContext(MLContextOptions* option);
  MLContext* createContext(GPUDevice* device);

  // WebNN Interface
  WNNInstance GetInstance() const;

 private:
  Member<ExecutionContext> execution_context_;
  // WebNN Implementation
  std::unique_ptr<WebnnInstance> webnn_instance_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_

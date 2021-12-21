// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml.h"

#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_instance.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

ML::ML(ExecutionContext* execution_context)
    : ExecutionContextLifecycleObserver(execution_context),
      execution_context_(execution_context) {
  webnn_instance_ = std::make_unique<WebnnInstance>();
  webnn_instance_->Initialize(execution_context_);
}

void ML::CreateModelLoader(ScriptState* script_state,
                           ExceptionState& exception_state) {
  NOTREACHED() << "Not Implemented yet";
}

void ML::Trace(Visitor* visitor) const {
  visitor->Trace(execution_context_);
  ScriptWrappable::Trace(visitor);
  ExecutionContextLifecycleObserver::Trace(visitor);
}

MLContext* ML::createContext(MLContextOptions* option) {
  // ScriptPromiseResolver* resolver =
  //     MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  // auto promise = resolver->Promise();

  // ExecutionContext* execution_context = ExecutionContext::From(script_state);
  // WebNN implementation
  WNNContext webnn_context = webnn_instance_->CreateContext(option);
  if (!webnn_context) {
    // resolver->Resolve(v8::Null(script_state->GetIsolate()));
    return nullptr;
  }

  // Notice that currently, we just create the context in the renderer. In the
  // future we may add backend query ability to check whether a context is
  // supportable or not. At that time, this function will be truly asynced.
  auto* ml_context = MakeGarbageCollected<MLContext>(
      option->devicePreference(), option->powerPreference(),
      option->modelFormat(), option->numThreads(), this, execution_context_,
      webnn_instance_->GetWebnnControlClient(), webnn_context);
  // resolver->Resolve(ml_context);

  return ml_context;
}

void ML::ContextDestroyed() {
  webnn_instance_->ContextDestroyed();
}

MLContext* ML::createContext(GPUDevice* device) {
  gpu::CommandBufferId command_buffer_id = device->GetCommandBufferID();
  std::tuple<uint32_t, uint32_t> device_id = device->GetDeviceID();
  WNNContext webnn_context = webnn_instance_->CreateContext(
      std::get<0>(device_id), std::get<1>(device_id), command_buffer_id);
  if (!webnn_context) {
    return nullptr;
  }

  // Notice that currently, we just create the context in the renderer. In the
  // future we may add backend query ability to check whether a context is
  // supportable or not. At that time, this function will be truly asynced.
  auto* ml_context = MakeGarbageCollected<MLContext>(
      this, execution_context_, webnn_instance_->GetWebnnControlClient(),
      webnn_context);

  return ml_context;
}

WNNInstance ML::GetInstance() const {
  return webnn_instance_->GetInstance();
}

}  // namespace blink

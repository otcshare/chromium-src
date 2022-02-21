// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml.h"

#include "third_party/blink/public/common/browser_interface_broker_proxy.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/frame/local_frame.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_instance.h"

namespace blink {

using ml::mojom::blink::LoadModelOptionsPtr;

ML::ML(ExecutionContext* execution_context)
    : ExecutionContextLifecycleObserver(execution_context),
      execution_context_(execution_context),
      remote_service_(execution_context_.Get()) {
  webnn_instance_ = std::make_unique<WebnnInstance>();
}

void ML::Load(ScriptState* script_state,
              Vector<uint8_t>& model_content,
              LoadModelOptionsPtr options,
              ExceptionState& exception_state,
              ml::mojom::blink::MLService::LoadCallback callback) {
  if (!BootstrapMojoConnectionIfNeeded(script_state, exception_state)) {
    return;
  }

  remote_service_->Load(std::move(model_content), std::move(options),
                        std::move(callback));
}

void ML::Trace(Visitor* visitor) const {
  visitor->Trace(execution_context_);
  visitor->Trace(remote_service_);

  ExecutionContextLifecycleObserver::Trace(visitor);
  ScriptWrappable::Trace(visitor);
}

void ML::ContextDestroyed() {
  webnn_instance_->ContextDestroyed();
}

MLContext* ML::createContext(MLContextOptions* option) {
  // ScriptPromiseResolver* resolver =
  //     MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  // auto promise = resolver->Promise();

  // ExecutionContext* execution_context = ExecutionContext::From(script_state);
  // WebNN implementation
  WNNContext webnn_context =
      webnn_instance_->CreateContext(execution_context_, option);
  if (!webnn_context) {
    // resolver->Resolve(v8::Null(script_state->GetIsolate()));
    return nullptr;
  }

  // Notice that currently, we just create the context in the renderer. In the
  // future we may add backend query ability to check whether a context is
  // supportable or not. At that time, this function will be truly asynced.
  auto* ml_context = MakeGarbageCollected<MLContext>(
      execution_context_, webnn_instance_->GetWebnnControlClient(),
      webnn_context, option->modelFormat(), this);
  // resolver->Resolve(ml_context);

  return ml_context;
}

WNNInstance ML::GetInstance() const {
  return webnn_instance_->GetInstance();
}

bool ML::BootstrapMojoConnectionIfNeeded(ScriptState* script_state,
                                         ExceptionState& exception_state) {
  // We need to do the following check because the execution context of this
  // navigator may be invalid (e.g. the frame is detached).
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "The execution context is invalid");
    return false;
  }
  // Note that we do not use `ExecutionContext::From(script_state)` because
  // the ScriptState passed in may not be guaranteed to match the execution
  // context associated with this navigator, especially with
  // cross-browsing-context calls.
  if (!remote_service_.is_bound()) {
    execution_context_->GetBrowserInterfaceBroker().GetInterface(
        remote_service_.BindNewPipeAndPassReceiver(
            execution_context_->GetTaskRunner(TaskType::kInternalDefault)));
  }
  return true;
}

}  // namespace blink

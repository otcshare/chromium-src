// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_context.h"

#include "gpu/command_buffer/client/webnn_interface.h"
#include "third_party/blink/public/platform/platform.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_uncaptured_error_event_init.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_mloutofmemoryerror_mlvalidationerror.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/inspector/console_message.h"
#include "third_party/blink/renderer/modules/event_target_modules.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_out_of_memory_error.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_uncaptured_error_event.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_validation_error.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_enum_conversions.h"
#include "third_party/blink/renderer/platform/bindings/microtask.h"

namespace blink {

WebnnContext::WebnnContext(
    ExecutionContext* execution_context,
    scoped_refptr<WebnnControlClientHolder> webnn_control_client,
    WNNContext webnn_context)
    : ExecutionContextClient(execution_context),
      WebnnObject(webnn_control_client, webnn_context),
      error_callback_(
          BindDawnRepeatingCallback(&WebnnContext::OnUncapturedError,
                                    WrapWeakPersistent(this))) {
  DCHECK(webnn_context);

  GetProcs().contextSetUncapturedErrorCallback(
      GetHandle(), error_callback_->UnboundCallback(),
      error_callback_->AsUserdata());
}

WebnnContext::~WebnnContext() {
  // Clear the callbacks since we can't handle callbacks after finalization.
  // error_callback_, logging_callback_, and lost_callback_ will be deleted.
  GetProcs().contextSetUncapturedErrorCallback(GetHandle(), nullptr, nullptr);
}

void WebnnContext::InjectError(WNNErrorType type, const char* message) {
  GetProcs().contextInjectError(GetHandle(), type, message);
}

void WebnnContext::AddConsoleWarning(const char* message) {
  ExecutionContext* execution_context = GetExecutionContext();
  if (execution_context && allowed_console_warnings_remaining_ > 0) {
    auto* console_message = MakeGarbageCollected<ConsoleMessage>(
        mojom::blink::ConsoleMessageSource::kRendering,
        mojom::blink::ConsoleMessageLevel::kWarning, message);
    execution_context->AddConsoleMessage(console_message);

    allowed_console_warnings_remaining_--;
    if (allowed_console_warnings_remaining_ == 0) {
      auto* final_message = MakeGarbageCollected<ConsoleMessage>(
          mojom::blink::ConsoleMessageSource::kRendering,
          mojom::blink::ConsoleMessageLevel::kWarning,
          "WebNN: too many warnings, no more warnings will be reported to the "
          "console for this WebnnContext.");
      execution_context->AddConsoleMessage(final_message);
    }
  }
}

void WebnnContext::OnUncapturedError(WNNErrorType errorType,
                                     const char* message) {
  DCHECK_NE(errorType, WNNErrorType_NoError);
  DCHECK_NE(errorType, WNNErrorType_DeviceLost);
  LOG(ERROR) << "WebnnContext: " << message;
  AddConsoleWarning(message);

  MLUncapturedErrorEventInit* init = MLUncapturedErrorEventInit::Create();
  if (errorType == WNNErrorType_Validation) {
    init->setError(
        MakeGarbageCollected<V8UnionMLOutOfMemoryErrorOrMLValidationError>(
            MakeGarbageCollected<MLValidationError>(message)));
  } else if (errorType == WNNErrorType_OutOfMemory) {
    init->setError(
        MakeGarbageCollected<V8UnionMLOutOfMemoryErrorOrMLValidationError>(
            MLOutOfMemoryError::Create()));
  } else {
    return;
  }
  DispatchEvent(*MLUncapturedErrorEvent::Create(
      event_type_names::kUncapturederror, init));
}

void WebnnContext::pushErrorScope(const WTF::String& filter) {
  GetProcs().contextPushErrorScope(GetHandle(),
                                   AsWebnnEnum<WNNErrorFilter>(filter));
}

ScriptPromise WebnnContext::popErrorScope(ScriptState* script_state) {
  ScriptPromiseResolver* resolver =
      MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();

  auto* callback =
      BindDawnOnceCallback(&WebnnContext::OnPopErrorScopeCallback,
                           WrapPersistent(this), WrapPersistent(resolver));

  if (!GetProcs().contextPopErrorScope(GetHandle(), callback->UnboundCallback(),
                                       callback->AsUserdata())) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kOperationError, "No error scopes to pop."));
    delete callback;
    return promise;
  }

  // WebNN guarantees that promises are resolved in finite time so we
  // need to ensure commands are flushed.
  EnsureFlush();
  return promise;
}

void WebnnContext::OnPopErrorScopeCallback(ScriptPromiseResolver* resolver,
                                           WNNErrorType type,
                                           const char* message) {
  v8::Isolate* isolate = resolver->GetScriptState()->GetIsolate();
  switch (type) {
    case WNNErrorType_NoError:
      resolver->Resolve(v8::Null(isolate));
      break;
    case WNNErrorType_OutOfMemory:
      resolver->Resolve(MLOutOfMemoryError::Create());
      break;
    case WNNErrorType_Validation:
      resolver->Resolve(MakeGarbageCollected<MLValidationError>(message));
      break;
    case WNNErrorType_Unknown:
    case WNNErrorType_DeviceLost:
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError));
      break;
    default:
      NOTREACHED();
  }
}

ExecutionContext* WebnnContext::GetExecutionContext() const {
  return ExecutionContextClient::GetExecutionContext();
}

const AtomicString& WebnnContext::InterfaceName() const {
  return event_target_names::kMLContext;
}

void WebnnContext::Trace(Visitor* visitor) const {
  ExecutionContextClient::Trace(visitor);
  EventTargetWithInlineData::Trace(visitor);
}

}  // namespace blink

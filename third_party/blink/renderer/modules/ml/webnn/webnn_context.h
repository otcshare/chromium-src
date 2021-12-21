// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CONTEXT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CONTEXT_H_

#include "base/memory/scoped_refptr.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_property.h"
#include "third_party/blink/renderer/core/dom/events/event_target.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"
#include "third_party/blink/renderer/platform/graphics/gpu/dawn_callback.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"

namespace blink {

class ExecutionContext;
class HTMLCanvasElement;
class ScriptPromiseResolver;
class ScriptState;

class WebnnContext : public EventTargetWithInlineData,
                     public ExecutionContextClient,
                     public WebnnObject<WNNContext> {
 public:
  explicit WebnnContext(
      ExecutionContext* execution_context,
      scoped_refptr<WebnnControlClientHolder> webnn_control_client,
      WNNContext webnn_context);

  WebnnContext(const WebnnContext&) = delete;
  WebnnContext& operator=(const WebnnContext&) = delete;

  ~WebnnContext() override;

  void Trace(Visitor* visitor) const override;

  void pushErrorScope(const WTF::String& filter);
  ScriptPromise popErrorScope(ScriptState* script_state);

  DEFINE_ATTRIBUTE_EVENT_LISTENER(uncapturederror, kUncapturederror)

  // EventTarget overrides.
  const AtomicString& InterfaceName() const override;
  ExecutionContext* GetExecutionContext() const override;

  void InjectError(WNNErrorType type, const char* message);
  void AddConsoleWarning(const char* message);

 private:
  void OnUncapturedError(WNNErrorType errorType, const char* message);
  void OnPopErrorScopeCallback(ScriptPromiseResolver* resolver,
                               WNNErrorType type,
                               const char* message);

  std::unique_ptr<DawnRepeatingCallback<void(WNNErrorType, const char*)>>
      error_callback_;

  static constexpr int kMaxAllowedConsoleWarnings = 500;
  int allowed_console_warnings_remaining_ = kMaxAllowedConsoleWarnings;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CONTEXT_H_

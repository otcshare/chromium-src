// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_

#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/public/mojom/ml/ml.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/frame/navigator.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"

namespace blink {

class MLContextOptions;
class ScriptPromise;
class ScriptState;

class ML final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit ML(ExecutionContext* execution_context);

  ML(const ML&) = delete;
  ML& operator=(const ML&) = delete;

  // Exposes the `Load` mojo interface of `remote_service_`. Will try to bind it
  // first.
  void Load(ScriptState* script_state,
            Vector<uint8_t>& model_content,
            ml::mojom::blink::LoadModelOptionsPtr options,
            ExceptionState& exception_state,
            ml::mojom::blink::MLService::LoadCallback callback);

  void Trace(blink::Visitor*) const override;

  // IDL interface:
  ScriptPromise createContext(ScriptState* state,
                              MLContextOptions* option,
                              ExceptionState& exception_state);

 private:
  // Bind the Mojo connection to browser process if needed.
  // Returns false when the execution context is not valid (e.g., the frame is
  // detached) and an exception will be thrown.
  // Otherwise returns true.
  bool BootstrapMojoConnectionIfNeeded(ScriptState*, ExceptionState&);

  Member<ExecutionContext> execution_context_;

  HeapMojoRemote<ml::mojom::blink::MLService> remote_service_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_

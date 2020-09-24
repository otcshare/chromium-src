// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/nn_compilation.h"

#include <utility>

#include "mojo/public/cpp/bindings/interface_ptr.h"
#include "services/ml/public/mojom/constants.mojom-blink.h"
#include "services/service_manager/public/cpp/interface_provider.h"
#include "third_party/blink/public/platform/web_string.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/neural_network_context.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_execution.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

namespace blink {

NNCompilation::NNCompilation(ml::mojom::blink::CompilationPtrInfo info,
                             int32_t preference,
                             HashMap<WTF::String, uint32_t> name_index)
    : name_index_(std::move(name_index)) {
  compilation_.Bind(std::move(info));
  compilation_.set_connection_error_handler(
      WTF::Bind(&NNCompilation::OnConnectionError, WrapWeakPersistent(this)));
  // Mojom interface
  compilation_->Finish(preference, WTF::Bind(&NNCompilation::OnResultCode,
                                             WrapPersistent(this)));
}

NNCompilation::~NNCompilation() = default;

void NNCompilation::OnResultCode(int32_t result_code) {
  LOG(INFO) << "The result code is " << result_code
            << "after finish creating compilation.";
}

ScriptPromise NNCompilation::createExecution(ScriptState* script_state) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();
  if (!compilation_) {
    resolver->Reject(
        MakeGarbageCollected<DOMException>(DOMExceptionCode::kNotSupportedError,
                                           "Compilation service unavailable."));
    return promise;
  }

  requests_.insert(resolver);

  compilation_->CreateExecution(WTF::Bind(&NNCompilation::OnCreateExecution,
                                          WrapPersistent(this),
                                          WrapPersistent(resolver)));
  return promise;
}

void NNCompilation::Trace(blink::Visitor* visitor) const {
  visitor->Trace(requests_);
  ScriptWrappable::Trace(visitor);
}

void NNCompilation::OnCreateExecution(
    ScriptPromiseResolver* resolver,
    int32_t result_code,
    ml::mojom::blink::ExecutionInitParamsPtr init_params) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code == ml::mojom::blink::NOT_ERROR) {
    resolver->Resolve(MakeGarbageCollected<NNExecution>(
        std::move(init_params), name_index_));
  } else {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError,
        "createExecution fails: " + String::Number(result_code)));
  }
}

void NNCompilation::OnConnectionError() {
  for (const auto& request : requests_) {
    request->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError,
        "Compilation V2 is not implemented."));
  }

  requests_.clear();
  compilation_.reset();
}

}  // namespace blink

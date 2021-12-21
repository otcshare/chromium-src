// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_model_loaded.h"

#include "third_party/abseil-cpp/absl/types/optional.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/frame/local_frame.h"
#include "third_party/blink/renderer/modules/ml/ml_type_converters.h"
#include "third_party/blink/renderer/platform/runtime_enabled_features.h"
#include "third_party/blink/renderer/platform/wtf/hash_map.h"

namespace blink {

using ml::mojom::blink::TensorPtr;

namespace {

void OnComputeResult(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    ml::mojom::blink::ComputeResult result,
    absl::optional<HashMap<String, TensorPtr>> outputs) {
  if (result != ml::mojom::blink::ComputeResult::kOk || !outputs.has_value()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kUnknownError, "Internal error."));
  }

  HeapVector<std::pair<String, Member<MLTensor>>> outputs_blink;
  for (const auto& output_mojo : outputs.value()) {
    outputs_blink.emplace_back(
        output_mojo.key, mojo::ConvertTo<MLTensor*>(output_mojo.value));
  }

  resolver->Resolve(std::move(outputs_blink));
}

}  // namespace

MLModelLoaded::MLModelLoaded(
    ExecutionContext* context,
    mojo::PendingRemote<ml::mojom::blink::ModelLoaded> pending_remote)
    : remote_service_(context) {
  remote_service_.Bind(std::move(pending_remote),
                       context->GetTaskRunner(TaskType::kInternalDefault));
}

MLModelLoaded::~MLModelLoaded() = default;

ScriptPromise MLModelLoaded::compute(
    ScriptState* script_state,
    const HeapVector<std::pair<String, Member<MLTensor>>>& inputs) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();

  HashMap<String, TensorPtr> input_mojo;
  for (const auto& name_tensor : inputs) {
    input_mojo.insert(
        name_tensor.first,
        mojo::ConvertTo<ml::mojom::blink::TensorPtr>(name_tensor.second.Get()));
  }

  remote_service_->Compute(
      std::move(input_mojo),
      WTF::Bind(&OnComputeResult, WrapPersistent(script_state),
                WrapPersistent(resolver)));

  return promise;
}

void MLModelLoaded::Trace(Visitor* visitor) const {
  visitor->Trace(remote_service_);

  ScriptWrappable::Trace(visitor);
}

}  // namespace blink

// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_MODEL_LOADED_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_MODEL_LOADED_H_

#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/public/mojom/ml/ml.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_typedefs.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

class ExecutionContext;
class ScriptState;

class MLModelLoaded final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  MLModelLoaded(
      ExecutionContext* context,
      mojo::PendingRemote<ml::mojom::blink::ModelLoaded> pending_remote);

  ~MLModelLoaded() override;

  // IDL Interface:
  ScriptPromise compute(
      ScriptState* script_state,
      const HeapVector<std::pair<String, Member<MLTensor>>>& inputs);

  void Trace(Visitor* visitor) const override;

 private:
  HeapMojoRemote<ml::mojom::blink::ModelLoaded> remote_service_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_MODEL_LOADED_H_

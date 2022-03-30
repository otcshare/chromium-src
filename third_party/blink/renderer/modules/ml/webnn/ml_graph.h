// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_GRAPH_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_GRAPH_H_

#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/core/typed_arrays/array_buffer_view_helpers.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_buffer.h"

namespace blink {

class MLContext;
class MLInput;
class MLBufferResourceView;
class ScriptPromiseResolver;
class ScriptState;
class ExceptionState;

typedef HeapVector<std::pair<String, Member<MLInputResource>>> MLNamedInputs;
typedef HeapVector<std::pair<String, Member<MLResource>>> MLNamedOutputs;

class MLGraph : public WebnnObject<WNNGraph> {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit MLGraph(MLContext* context, WNNGraph operand);

  MLGraph(const MLGraph&) = delete;
  MLGraph& operator=(const MLGraph&) = delete;

  void Trace(Visitor* visitor) const override;

  // ml_graph.idl
  void compute(const MLNamedInputs& inputs, const MLNamedOutputs& outputs);
  ScriptPromise computeAsync(ScriptState* script_state,
                             const MLNamedInputs& inputs,
                             const MLNamedOutputs& outputs,
                             ExceptionState& exception_state);

 private:
  void OnComputeAsyncCallback(ScriptPromiseResolver* resolver,
                              WNNComputeGraphStatus status,
                              const char* message);
  WNNNamedInputs CreateAndPopulateNamedInputs(const MLNamedInputs& inputs);
  WNNNamedOutputs CreateAndPopulateNamedOutputs(const MLNamedOutputs& outputs);
  HeapVector<Member<GPUBuffer>> gpu_buffers_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_GRAPH_H_

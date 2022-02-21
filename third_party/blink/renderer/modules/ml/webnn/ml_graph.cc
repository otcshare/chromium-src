// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_buffer_resource_view.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_arraybufferview_mlbufferresourceview.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_arraybufferview_mlbufferresourceview_mlinput.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"

namespace blink {

MLGraph::MLGraph(MLContext* context, WNNGraph graph)
    : WebnnObject<WNNGraph>(context, graph) {}

void MLGraph::Trace(Visitor* visitor) const {
  WebnnObject<WNNGraph>::Trace(visitor);
}

ScriptPromise MLGraph::computeAsync(ScriptState* script_state,
                                    const MLNamedInputs& inputs,
                                    const MLNamedOutputs& outputs,
                                    ExceptionState& exception_state) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();

  WNNNamedInputs webnn_inputs = CreateAndPopulateNamedInputs(inputs);
  WNNNamedOutputs webnn_outputs = CreateAndPopulateNamedOutputs(outputs);
  auto* callback =
      BindDawnOnceCallback(&MLGraph::OnComputeAsyncCallback,
                           WrapPersistent(this), WrapPersistent(resolver));
  GetProcs().graphComputeAsync(GetHandle(), webnn_inputs, webnn_outputs,
                               callback->UnboundCallback(),
                               callback->AsUserdata());
  FlushNow();

  return promise;
}

void MLGraph::OnComputeAsyncCallback(ScriptPromiseResolver* resolver,
                                     WNNComputeGraphStatus status,
                                     const char* message) {
  switch (status) {
    case WNNComputeGraphStatus_Success: {
      resolver->Resolve();
      break;
    }

    case WNNComputeGraphStatus_Error:
    case WNNComputeGraphStatus_ContextLost:
    case WNNComputeGraphStatus_Unknown: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError, message));
      break;
    }

    default: {
      NOTREACHED();
    }
  }
}

WNNNamedInputs MLGraph::CreateAndPopulateNamedInputs(
    const MLNamedInputs& inputs) {
  // Get WebNNInstance
  WNNInstance instance = context_.Get()->GetML()->GetInstance();
  WNNNamedInputs webnn_inputs = GetProcs().instanceCreateNamedInputs(instance);
  for (wtf_size_t i = 0; i < inputs.size(); ++i) {
    std::string name = inputs[i].first.Utf8();
    WNNInput webnn_input = AsWebnnType(inputs[i].second.Get());
    GetProcs().namedInputsSet(webnn_inputs, name.c_str(), &webnn_input);
  }
  return webnn_inputs;
}

WNNNamedOutputs MLGraph::CreateAndPopulateNamedOutputs(
    const MLNamedOutputs& outputs) {
  // Get WebNNInstance
  WNNInstance instance = context_.Get()->GetML()->GetInstance();
  WNNNamedOutputs webnn_outputs =
      GetProcs().instanceCreateNamedOutputs(instance);
  for (wtf_size_t i = 0; i < outputs.size(); ++i) {
    std::string name = outputs[i].first.Utf8();
    WNNArrayBufferView webnn_output = AsWebnnType(outputs[i].second);
    GetProcs().namedOutputsSet(webnn_outputs, name.c_str(), &webnn_output);
  }
  return webnn_outputs;
}

}  // namespace blink

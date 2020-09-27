// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_MODEL_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_MODEL_H_

#include "services/ml/public/mojom/model.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/v2/compilation_options.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_context.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

namespace blink {

int32_t StringToOperandType(const String& operand_type);
using NamedOperandVector = HeapVector<Member<NamedOperand>>;

class NNModel final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  NNModel(ml::mojom::blink::ModelPtrInfo, NamedOperandVector*);
  ~NNModel() override;

  ScriptPromise createCompilation(ScriptState*,
                                  const CompilationOptions* options);
  void BuildNeuralNetworkModel(Operand*);
  void AddScalarOperand(uint32_t index, int value);
  void AddScalarOperand(uint32_t index, float value);
  void AddScalarOperand(uint32_t index, bool value);
  void AddBiasOperand(uint32_t index, uint32_t output_channel);
  void AddTensorOperand(uint32_t index,
                        Vector<uint32_t> dimensions,
                        Vector<int32_t> value);
  void AddUnspecifiedOperand();
  void AddOperand(const OperandDescriptor* descriptor);
  void SetOperandValue(uint32_t index, DOMArrayBufferView* data);
  void AddOperation(int32_t type,
                    const Vector<uint32_t>& inputs,
                    const Vector<uint32_t>& outputs);
  void IdentifyInput(const String& name, uint32_t index);
  void IdentifyOutput(const String& name, uint32_t index);

  void Trace(blink::Visitor*) const override;

 private:
  void FinishCreatingModel(NamedOperandVector* outputs);
  void OnResultCode(int32_t);
  void OnCreateCompilation(ScriptPromiseResolver*,
                           int32_t preference,
                           int32_t,
                           ml::mojom::blink::CompilationInitParamsPtr);
  void OnConnectionError();

  ml::mojom::blink::ModelPtr model_;
  ml::mojom::blink::ModelInfoPtr model_info_;
  HeapHashSet<Member<ScriptPromiseResolver>> requests_;
  HeapHashMap<WTF::String, Member<DOMArrayBufferView>> buffer_views_;
  // name_index_ is used for getting index for setInput/setOutput in execution
  // phase, the index is sequence start from 0.
  HashMap<WTF::String, uint32_t> name_index_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_MODEL_H_

// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_CONTEXT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_CONTEXT_H_

#include "services/ml/public/mojom/neuralnetwork.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/execution_context/execution_context_lifecycle_observer.h"
#include "third_party/blink/renderer/modules/ml/v2/named_operand.h"
#include "third_party/blink/renderer/modules/ml/v2/operand.h"
#include "third_party/blink/renderer/modules/ml/v2/operand_descriptor.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/thread_state.h"
#include "third_party/blink/renderer/platform/wtf/text/atomic_string.h"

namespace blink {

class Model;
class NavigatorML;

using NamedOperandVector = HeapVector<Member<NamedOperand>>;

int32_t product(const WTF::Vector<int32_t>&);

class NNContext final : public ScriptWrappable,
                        public ExecutionContextLifecycleObserver {
  DEFINE_WRAPPERTYPEINFO();
  USING_GARBAGE_COLLECTED_MIXIN(NNContext);
  USING_PRE_FINALIZER(NNContext, Dispose);

 public:
  explicit NNContext(NavigatorML*);
  ~NNContext() override;

  Operand* input(const String&, const OperandDescriptor*, ExceptionState&);
  Operand* constant(const OperandDescriptor*,
                    MaybeShared<DOMArrayBufferView>,
                    ExceptionState&);
  Operand* add(Operand*, Operand*);
  Operand* mul(Operand*, Operand*);
  Operand* conv2d(Operand*,
                  Operand*,
                  WTF::Vector<int32_t>,
                  WTF::Vector<int32_t>,
                  WTF::Vector<int32_t>,
                  int32_t,
                  String,
                  ExceptionState&);
  Operand* averagePool2d(Operand*,
                         WTF::Vector<int32_t>,
                         WTF::Vector<int32_t>,
                         WTF::Vector<int32_t>,
                         WTF::Vector<int32_t>,
                         String,
                         ExceptionState&);
  Operand* maxPool2d(Operand*,
                     WTF::Vector<int32_t>,
                     WTF::Vector<int32_t>,
                     WTF::Vector<int32_t>,
                     WTF::Vector<int32_t>,
                     String,
                     ExceptionState&);
  Operand* reshape(Operand*, WTF::Vector<int32_t>);
  Operand* softmax(Operand*);
  Operand* relu(Operand*);
  ScriptPromise createModel(ScriptState*, const NamedOperandVector&);

  // ExecutionContextLifecycleObserver overrides.
  void ContextDestroyed() override;

  // Interface required by garbage collection.
  void Trace(Visitor*) const override;

 private:
  void OnCreateModel(ScriptPromiseResolver*,
                     NamedOperandVector*,
                     int32_t,
                     ml::mojom::blink::ModelInitParamsPtr);
  void OnConnectionError();
  void Dispose();

 private:
  ml::mojom::blink::NeuralNetworkPtr neural_network_;
  HeapHashSet<Member<ScriptPromiseResolver>> requests_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_CONTEXT_H_

// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include "third_party/blink/renderer/modules/ml/v2/nn_model.h"

#include <utility>

#include "mojo/public/cpp/bindings/interface_ptr.h"
#include "services/ml/public/mojom/constants.mojom-blink.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_typed_array.h"
#include "third_party/blink/renderer/modules/ml/neural_network_context.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_compilation.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_context.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

// Utils functions
int32_t StringToOperandType(const String& operand_type) {
  if (operand_type == "float32") {
    return NeuralNetworkContext::kFloat32;
  } else if (operand_type == "int32") {
    return NeuralNetworkContext::kInt32;
  } else if (operand_type == "uint32") {
    return NeuralNetworkContext::kUint32;
  } else if (operand_type == "tensor-float32") {
    return NeuralNetworkContext::kTensorFloat32;
  } else if (operand_type == "tensor-int32") {
    return NeuralNetworkContext::kTensorInt32;
  } else if (operand_type == "tensor-quant8-asymm") {
    return NeuralNetworkContext::kTensorQuant8Asymm;
  } else {
    // float16 and tensor-float16 are not supported in Android NN API design.
    NOTREACHED();
  }
  return 100;
}

namespace {

void CopyDataToSharedBuffer(
    const HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
    ml::mojom::blink::ModelInfoPtr& model_info) {
  uint32_t total_byte_length = 0;
  for (HeapHashMap<WTF::String, Member<DOMArrayBufferView>>::const_iterator
           itr = buffer_views.begin();
       itr != buffer_views.end(); ++itr) {
    total_byte_length += itr->value->deprecatedByteLengthAsUnsigned();
  }

  mojo::ScopedSharedBufferHandle memory;
  mojo::ScopedSharedBufferMapping mapping;
  if (total_byte_length != 0) {
    memory = mojo::SharedBufferHandle::Create(total_byte_length);
    mapping = memory->Map(total_byte_length);
  }

  uint32_t offset = 0;
  for (WTF::HashMap<WTF::String,
                    ml::mojom::blink::OperandValueInfoPtr>::const_iterator itr =
           model_info->values.begin();
       itr != model_info->values.end(); ++itr) {
    const ml::mojom::blink::OperandValueInfoPtr& value_info = itr->value;
    DOMArrayBufferView* view =
        buffer_views.at(WTF::String::Number(value_info->index));
    uint32_t length = view->deprecatedByteLengthAsUnsigned();
    value_info->offset = offset;
    value_info->length = length;
    uint8_t* base = static_cast<uint8_t*>(mapping.get()) + offset;
    memcpy(static_cast<void*>(base), view->BaseAddress(), length);
    offset += length;
  }

  if (total_byte_length != 0) {
    model_info->memory =
        memory->Clone(mojo::SharedBufferHandle::AccessMode::READ_ONLY);
    model_info->memory_size = total_byte_length;
  }
}

}  // namespace

NNModel::NNModel(ml::mojom::blink::ModelPtrInfo info,
                 NamedOperandVector* outputs) {
  model_.Bind(std::move(info));
  model_.set_connection_error_handler(
      WTF::Bind(&NNModel::OnConnectionError, WrapWeakPersistent(this)));
  FinishCreatingModel(std::move(outputs));
}

NNModel::~NNModel() = default;

// Traversal graph inorder to create model.
void NNModel::BuildNeuralNetworkModel(Operand* root) {
  // The index is sequence increase to index all operands in model.
  uint32_t index = 0;
  // the stack is used for traversaling model tree.
  HeapVector<Member<Operand>> stack;
  Operand* operand = root;
  while (operand || !stack.IsEmpty()) {
    while (operand) {
      stack.push_back(operand);
      operand = operand->FirstInput();
    }
    // It will be input/constant if there is no FirstInput operand.
    if (!stack.IsEmpty()) {
      // The index_ in Operand will be add when calling NextInput, so keep the
      // line.
      Operand* next_input = stack.back()->NextInput();
      if (next_input && !next_input->Traversal()) {
        operand = next_input;
      } else {
        // Call the AddLayer virtual function to Add the operand with Android NN
        // API.
        stack.back()->AddLayer(this, index);
        // Set the operand as traversalled so that it doesn't push again.
        stack.back()->SetTraversal(true);
        // Pop the node.
        stack.pop_back();
        // Don't use the code [operand = stack.back()] so that Origin branch
        // [while (operand)] will be not traversalled again.
        // operand = stack.back();
      }
    }
  }
}

void NNModel::FinishCreatingModel(NamedOperandVector* outputs) {
  model_info_ = ml::mojom::blink::ModelInfo::New();
  for (uint32_t i = 0; i < outputs->size(); ++i) {
    NamedOperand* output = (*outputs)[i];
    BuildNeuralNetworkModel(output->operand());
    // identify output and map name and index that will be used in execution.
    IdentifyOutput(output->name(), output->operand()->Index());
  }
  CopyDataToSharedBuffer(buffer_views_, model_info_);

  model_->Finish(std::move(model_info_),
                 WTF::Bind(&NNModel::OnResultCode, WrapPersistent(this)));
  return;
}

// Types not prefaced by ANEURALNETWORKS_TENSOR_* represent scalar values and
// must have no dimensions.
void NNModel::AddScalarOperand(uint32_t index, int value) {
  model_info_->operands.push_back(ml::mojom::blink::Operand::New(
      static_cast<int32_t>(NeuralNetworkContext::kInt32),
      WTF::Vector<uint32_t>(), 0, 0));

  // setOperandValue
  NotShared<DOMArrayBufferView> data =
      NotShared<DOMArrayBufferView>(DOMInt32Array::Create(&value, 1));
  SetOperandValue(index, data.View());
}

// Types prefaced with ANEURALNETWORKS_TENSOR_* must be used for tensor data
// (i.e., tensors with at least one dimension).
void NNModel::AddBiasOperand(uint32_t index, uint32_t output_channel) {
  model_info_->operands.push_back(ml::mojom::blink::Operand::New(
      static_cast<int32_t>(NeuralNetworkContext::kTensorFloat32),
      WTF::Vector<uint32_t>(1, output_channel), 0, 0));

  // setOperandValue
  Vector<float> value(output_channel, 0);
  NotShared<DOMArrayBufferView> data = NotShared<DOMArrayBufferView>(
      DOMFloat32Array::Create(value.data(), output_channel));
  SetOperandValue(index, data.View());
}

void NNModel::AddTensorOperand(uint32_t index,
                               Vector<uint32_t> dimensions,
                               Vector<int32_t> value) {
  model_info_->operands.push_back(ml::mojom::blink::Operand::New(
      static_cast<int32_t>(NeuralNetworkContext::kTensorInt32),
      std::move(dimensions), 0, 0));

  // setOperandValue
  NotShared<DOMArrayBufferView> data = NotShared<DOMArrayBufferView>(
      DOMInt32Array::Create(value.data(), value.size()));
  SetOperandValue(index, data.View());
}

void NNModel::AddUnspecifiedOperand() {
  model_info_->operands.push_back(ml::mojom::blink::Operand::New(
      static_cast<int32_t>(NeuralNetworkContext::kTensorFloat32),
      WTF::Vector<uint32_t>(), 0, 0));
}

void NNModel::AddOperand(const OperandDescriptor* descriptor) {
  model_info_->operands.push_back(ml::mojom::blink::Operand::New(
      StringToOperandType(descriptor->type()),
      descriptor->hasDimensions() ? descriptor->dimensions()
                                  : WTF::Vector<uint32_t>(),
      descriptor->hasScale() ? descriptor->scale() : 0,
      descriptor->hasZeroPoint() ? descriptor->zeroPoint() : 0));
}

void NNModel::SetOperandValue(uint32_t index, DOMArrayBufferView* data) {
  WTF::String index_str = WTF::String::Number(index);
  model_info_->values.insert(
      index_str, ml::mojom::blink::OperandValueInfo::New(index, 0, 0));
  buffer_views_.insert(index_str, data);
}

void NNModel::AddOperation(int32_t type,
                           const Vector<uint32_t>& inputs,
                           const Vector<uint32_t>& outputs) {
  model_info_->operations.push_back(
      ml::mojom::blink::Operation::New(type, inputs, outputs));
}

void NNModel::IdentifyInput(const String& name, uint32_t index) {
  name_index_.insert(name, model_info_->inputs.size());
  model_info_->inputs.push_back(index);
}

void NNModel::IdentifyOutput(const String& name, uint32_t index) {
  name_index_.insert(name, model_info_->outputs.size());
  model_info_->outputs.push_back(index);
}

void NNModel::OnResultCode(int32_t result_code) {
  LOG(INFO) << "The result code is " << result_code
            << "after finish creating model.";
}

ScriptPromise NNModel::createCompilation(ScriptState* script_state,
                                         const CompilationOptions* options) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();
  if (!model_) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError, "Model service unavailable."));
    return promise;
  }

  int32_t preference;
  const String& operand_type =
      options->hasPowerPreference() ? options->powerPreference() : "default";
  if (operand_type == "default") {
    preference = NeuralNetworkContext::kPreferFastSingleAnswer;
  } else if (operand_type == "high-performance") {
    preference = NeuralNetworkContext::kPreferSustainedSpeed;
  } else if (operand_type == "low-power") {
    preference = NeuralNetworkContext::kPreferLowPower;
  } else {
    resolver->Reject(
        MakeGarbageCollected<DOMException>(DOMExceptionCode::kNotSupportedError,
                                           "The preference isn't supported."));
    return promise;
  }

  requests_.insert(resolver);

  model_->CreateCompilation(WTF::Bind(&NNModel::OnCreateCompilation,
                                      WrapPersistent(this),
                                      WrapPersistent(resolver), preference));
  return promise;
}

void NNModel::OnCreateCompilation(
    ScriptPromiseResolver* resolver,
    int32_t preference,
    int32_t result_code,
    ml::mojom::blink::CompilationInitParamsPtr init_params) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code == ml::mojom::blink::NOT_ERROR) {
    resolver->Resolve(MakeGarbageCollected<NNCompilation>(
        std::move(init_params->compilation), preference,
        std::move(name_index_)));
  } else {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError,
        "createCompilation fails: " + String::Number(result_code)));
  }
}

void NNModel::Trace(blink::Visitor* visitor) const {
  visitor->Trace(requests_);
  visitor->Trace(buffer_views_);
  ScriptWrappable::Trace(visitor);
}

void NNModel::OnConnectionError() {
  for (const auto& request : requests_) {
    request->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError, "Model is not implemented."));
  }
  requests_.clear();
  model_.reset();
}

}  // namespace blink

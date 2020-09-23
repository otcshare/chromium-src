// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/nn_context.h"

#include <utility>

#include "services/ml/public/mojom/constants.mojom-blink.h"
#include "third_party/blink/public/common/browser_interface_broker_proxy.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/frame/local_dom_window.h"
#include "third_party/blink/renderer/modules/ml/navigator_ml.h"
#include "third_party/blink/renderer/modules/ml/neural_network_context.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_model.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/binary.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/constant.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/conv.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/input.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/pooling.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/relu.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/reshape.h"
#include "third_party/blink/renderer/modules/ml/v2/ops/softmax.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"

namespace blink {

int32_t product(const WTF::Vector<int32_t>& dims) {
  uint32_t prod = 1;
  for (auto dim : dims)
    prod *= dim;

  return prod;
}

namespace {

bool InvalidData(ExceptionState& state) {
  state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                          "Data type is invalid.");
  return true;
}

bool InvalidOperand(const OperandDescriptor* descriptor,
                    ExceptionState& state) {
  if (!descriptor->hasType())
    return InvalidData(state);

  switch (StringToOperandType(descriptor->type())) {
    case NeuralNetworkContext::kFloat32:
    case NeuralNetworkContext::kInt32:
    case NeuralNetworkContext::kUint32:
      if (descriptor->hasDimensions())
        return InvalidData(state);
      break;
    case NeuralNetworkContext::kTensorFloat32:
    case NeuralNetworkContext::kTensorInt32:
      if (!descriptor->hasDimensions())
        return InvalidData(state);
      break;
    case NeuralNetworkContext::kTensorQuant8Asymm:
      if (!descriptor->hasDimensions() || !descriptor->hasScale() ||
          !descriptor->hasZeroPoint() || descriptor->scale() < 0 ||
          descriptor->zeroPoint() < 0 || descriptor->zeroPoint() > 255)
        return InvalidData(state);
      break;
    default:
      NOTREACHED();
  }
  return false;
}

bool InvalidOperandValue(const OperandDescriptor* descriptor,
                         const DOMArrayBufferView* data,
                         ExceptionState& exception_state) {
  bool invalid = false;
  auto data_type = data->GetType();
  switch (StringToOperandType(descriptor->type())) {
    case NeuralNetworkContext::kFloat32:
      if (data->deprecatedByteLengthAsUnsigned() / data->TypeSize() > 1)
        invalid = true;
      FALLTHROUGH;
    case NeuralNetworkContext::kTensorFloat32:
      if (data_type != DOMArrayBufferView::kTypeFloat32)
        invalid = true;
      break;
    case NeuralNetworkContext::kInt32:
      if (data->deprecatedByteLengthAsUnsigned() / data->TypeSize() > 1)
        invalid = true;
      FALLTHROUGH;
    case NeuralNetworkContext::kTensorInt32:
      if (data_type != DOMArrayBufferView::kTypeInt32)
        invalid = true;
      break;
    case NeuralNetworkContext::kUint32:
      if (data_type != DOMArrayBufferView::kTypeUint32 ||
          data->deprecatedByteLengthAsUnsigned() / data->TypeSize() > 1)
        invalid = true;
      break;
    case NeuralNetworkContext::kTensorQuant8Asymm:
      if (data_type != DOMArrayBufferView::kTypeUint8)
        invalid = true;
      break;
    case NeuralNetworkContext::kTensorQuant8AsymmSigned:
      if (data_type != DOMArrayBufferView::kTypeInt8)
        invalid = true;
      break;
    case NeuralNetworkContext::kTensorQuant8SymmPerChannel:
      if (data_type != DOMArrayBufferView::kTypeInt8)
        invalid = true;
      break;
    default:
      NOTREACHED();
  }

  return invalid ? InvalidData(exception_state) : false;
}

bool InvalidStrides(WTF::Vector<int32_t>& padding,
                    WTF::Vector<int32_t>& strides,
                    WTF::Vector<int32_t>& dilations,
                    ExceptionState& state) {
  if (padding.IsEmpty()) {
    padding = Vector<int32_t>(4, 0);
  }
  if (strides.IsEmpty()) {
    strides = Vector<int32_t>(2, 1);
  }
  if (dilations.IsEmpty()) {
    dilations = Vector<int32_t>(2, 1);
  }
  if (product(padding) < 0 || product(strides) <= 0 ||
      product(dilations) <= 0) {
    return InvalidData(state);
  }
  return false;
}

}  // namespace

NNContext::NNContext(NavigatorML* navigator_ml)
    : ExecutionContextLifecycleObserver(navigator_ml->DomWindow()) {
  navigator_ml->DomWindow()->GetBrowserInterfaceBroker().GetInterface(
      mojo::MakeRequest(&neural_network_));
  neural_network_.set_connection_error_handler(
      WTF::Bind(&NNContext::OnConnectionError, WrapWeakPersistent(this)));
}

NNContext::~NNContext() = default;

Operand* NNContext::input(const String& name,
                          const OperandDescriptor* descriptor,
                          ExceptionState& state) {
  if (InvalidOperand(descriptor, state))
    return nullptr;
  return MakeGarbageCollected<Input>(name, descriptor);
}

Operand* NNContext::constant(const OperandDescriptor* descriptor,
                             MaybeShared<DOMArrayBufferView> data,
                             ExceptionState& state) {
  if (InvalidOperandValue(descriptor, data.View(), state))
    return nullptr;
  return MakeGarbageCollected<Constant>(descriptor, data.View());
}

Operand* NNContext::add(Operand* primary, Operand* secondary) {
  return MakeGarbageCollected<Binary>(kBinaryTypeAdd, primary, secondary);
}

Operand* NNContext::mul(Operand* primary, Operand* secondary) {
  return MakeGarbageCollected<Binary>(kBinaryTypeMul, primary, secondary);
}

Operand* NNContext::conv2d(Operand* input,
                           Operand* filter,
                           WTF::Vector<int32_t> padding,
                           WTF::Vector<int32_t> strides,
                           WTF::Vector<int32_t> dilations,
                           int32_t groups,
                           String layout,
                           ExceptionState& state) {
  if (InvalidStrides(padding, strides, dilations, state))
    return nullptr;
  return MakeGarbageCollected<Conv>(input, filter, std::move(padding),
                                    std::move(strides), std::move(dilations),
                                    groups, layout);
}

Operand* NNContext::averagePool2d(Operand* input,
                                  WTF::Vector<int32_t> window_dimensions,
                                  WTF::Vector<int32_t> padding,
                                  WTF::Vector<int32_t> strides,
                                  WTF::Vector<int32_t> dilations,
                                  String layout,
                                  ExceptionState& state) {
  if (InvalidStrides(padding, strides, dilations, state))
    return nullptr;
  if (window_dimensions.IsEmpty())
    window_dimensions = WTF::Vector<int32_t>(2, 0);
  return MakeGarbageCollected<Pooling>(
      input, std::move(window_dimensions), std::move(padding),
      std::move(strides), std::move(dilations), layout, kPoolingTypeAverage);
}

Operand* NNContext::maxPool2d(Operand* input,
                              WTF::Vector<int32_t> window_dimensions,
                              WTF::Vector<int32_t> padding,
                              WTF::Vector<int32_t> strides,
                              WTF::Vector<int32_t> dilations,
                              String layout,
                              ExceptionState& state) {
  if (InvalidStrides(padding, strides, dilations, state))
    return nullptr;
  if (window_dimensions.IsEmpty())
    window_dimensions = WTF::Vector<int32_t>(2, 0);
  return MakeGarbageCollected<Pooling>(
      input, std::move(window_dimensions), std::move(padding),
      std::move(strides), std::move(dilations), layout, kPoolingTypeMax);
}

Operand* NNContext::reshape(Operand* input, WTF::Vector<int32_t> new_shape) {
  return MakeGarbageCollected<Reshape>(input, std::move(new_shape));
}

Operand* NNContext::softmax(Operand* input) {
  return MakeGarbageCollected<Softmax>(input);
}

Operand* NNContext::relu(Operand* input) {
  return MakeGarbageCollected<Relu>(input);
}

ScriptPromise NNContext::createModel(ScriptState* script_state,
                                     const NamedOperandVector& outputs) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();
  if (!neural_network_) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError,
        "Neural Network service unavailable."));
    return promise;
  }

  requests_.insert(resolver);
  NamedOperandVector* operand_outputs =
      MakeGarbageCollected<NamedOperandVector>(outputs);
  neural_network_->CreateModel(
      WTF::Bind(&NNContext::OnCreateModel, WrapPersistent(this),
                WrapPersistent(resolver), WrapPersistent(operand_outputs)));
  return promise;
}

void NNContext::ContextDestroyed() {
  Dispose();
}

void NNContext::Trace(Visitor* visitor) const {
  visitor->Trace(requests_);
  ScriptWrappable::Trace(visitor);
  ExecutionContextLifecycleObserver::Trace(visitor);
}

void NNContext::OnCreateModel(
    ScriptPromiseResolver* resolver,
    NamedOperandVector* outputs,
    int32_t result_code,
    ml::mojom::blink::ModelInitParamsPtr init_params) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code == ml::mojom::blink::NOT_ERROR) {
    resolver->Resolve(MakeGarbageCollected<NNModel>(
        std::move(init_params->model), std::move(outputs)));
  } else {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError,
        "createModel fails: " + String::Number(result_code)));
  }
}

void NNContext::OnConnectionError() {
  for (const auto& request : requests_) {
    request->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError,
        "Neural Network is not implemented."));
  }
  requests_.clear();
  neural_network_.reset();
}

void NNContext::Dispose() {}

}  // namespace blink

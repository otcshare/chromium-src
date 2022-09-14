// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/mojo_graph.h"

#include "mojo/public/cpp/bindings/pending_remote.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_context.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

#include <memory>

namespace blink {

namespace {

using ml::webnn::mojom::blink::AutoPad;
using ml::webnn::mojom::blink::BinaryOperandType;
using ml::webnn::mojom::blink::BuildResult;
using ml::webnn::mojom::blink::ComputeResult;
using ml::webnn::mojom::blink::Conv2dFilterOperandLayout;
using ml::webnn::mojom::blink::FusionType;
using ml::webnn::mojom::blink::InputOperandLayout;
using ml::webnn::mojom::blink::MemoryInfoPtr;
using ml::webnn::mojom::blink::OperandType;
using ml::webnn::mojom::blink::Pool2dType;
using ml::webnn::mojom::blink::RoundingType;
using ml::webnn::mojom::blink::UnaryOperandType;

#if BUILDFLAG(IS_WIN)
static const uint32_t kBufferAlignment = 16;
#else
static const uint32_t kBufferAlignment = 1;
#endif
base::CheckedNumeric<size_t> Align(size_t value, uint32_t aligment) {
  size_t remainder = value % aligment;
  if (remainder != 0) {
    value += aligment - remainder;
  }

  return value;
}

OperandType BlinkOperandTypeToMojo(V8MLOperandType::Enum type) {
  switch (type) {
    case V8MLOperandType::Enum::kFloat32:
      return OperandType::kFloat32;
    case V8MLOperandType::Enum::kFloat16:
      return OperandType::kFloat16;
    case V8MLOperandType::Enum::kInt32:
      return OperandType::kInt32;
    case V8MLOperandType::Enum::kUint32:
      return OperandType::kUint32;
    case V8MLOperandType::Enum::kInt8:
      return OperandType::kInt8;
    case V8MLOperandType::Enum::kUint8:
      return OperandType::kUint8;
  }
}

InputOperandLayout BlinkInputOperandLayoutToMojo(
    V8MLInputOperandLayout::Enum type) {
  switch (type) {
    case V8MLInputOperandLayout::Enum::kNchw:
      return InputOperandLayout::kNchw;
    case V8MLInputOperandLayout::Enum::kNhwc:
      return InputOperandLayout::kNhwc;
  }
}

Conv2dFilterOperandLayout BlinkConv2dFilterOperandLayoutToMojo(
    V8MLConv2dFilterOperandLayout::Enum type) {
  switch (type) {
    case V8MLConv2dFilterOperandLayout::Enum::kOihw:
      return Conv2dFilterOperandLayout::kOihw;
    case V8MLConv2dFilterOperandLayout::Enum::kHwio:
      return Conv2dFilterOperandLayout::kHwio;
    case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
      return Conv2dFilterOperandLayout::kOhwi;
    case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
      return Conv2dFilterOperandLayout::kIhwo;
  }
}

AutoPad BlinkAutoPadToMojo(V8MLAutoPad::Enum type) {
  switch (type) {
    case V8MLAutoPad::Enum::kExplicit:
      return AutoPad::kExplicit;
    case V8MLAutoPad::Enum::kSameUpper:
      return AutoPad::kSameUpper;
    case V8MLAutoPad::Enum::kSameLower:
      return AutoPad::kSameLower;
  }
}

RoundingType BlinkRoundingTypeToMojo(V8MLRoundingType::Enum type) {
  switch (type) {
    case V8MLRoundingType::Enum::kFloor:
      return RoundingType::kFloor;
    case V8MLRoundingType::Enum::kCeil:
      return RoundingType::kCeil;
  }
}

Pool2dType BlinkPool2dTypeToMojo(MLOperator::OpKind type) {
  switch (type) {
    case MLOperator::OpKind::kAveragePool2d:
      return Pool2dType::kAveragePool2d;
    default:
      NOTREACHED();
      return Pool2dType::kUnknown;
  }
}

UnaryOperandType BlinkUnaryOperandTypeToMojo(MLOperator::OpKind type) {
  switch (type) {
    case MLOperator::OpKind::kRelu:
      return UnaryOperandType::kRelu;
    case MLOperator::OpKind::kSoftmax:
      return UnaryOperandType::kSoftmax;
    default:
      NOTREACHED();
      return UnaryOperandType::kUnknown;
  }
}

BinaryOperandType BlinkBinaryOperandTypeToMojo(MLOperator::OpKind type) {
  switch (type) {
    case MLOperator::OpKind::kAdd:
      return BinaryOperandType::kAdd;
    default:
      NOTREACHED();
      return BinaryOperandType::kUnknown;
  }
}

ml::webnn::mojom::blink::ClampOptionsPtr BlinkClampOptioinToMojo(
    const MLClampOptions* ml_options) {
  const float min = ml_options->hasMinValue()
                        ? ml_options->minValue()
                        : -std::numeric_limits<float>::infinity();
  const float max = ml_options->hasMaxValue()
                        ? ml_options->maxValue()
                        : +std::numeric_limits<float>::infinity();
  auto options = ml::webnn::mojom::blink::ClampOptions::New();
  options->minValue = min;
  options->maxValue = max;

  return options;
}

void AddClamp(const MLOperator* clamp,
              OperandDescriptorPtr output_desc,
              ml::webnn::mojom::blink::Graph* remote_graph) {
  const MLClampOptions* ml_options =
      static_cast<const MLClampOptions*>(clamp->Options());
  auto* input = clamp->Inputs()[0].Get();
  remote_graph->AddClamp(input->GetObjectId(),
                         BlinkClampOptioinToMojo(ml_options),
                         std::move(output_desc));
}

FusionOperatorPtr AddFusionOperator(
    const MLOperator* activation,
    ml::webnn::mojom::blink::Graph* remote_graph) {
  auto fusion_operation = ml::webnn::mojom::blink::FusionOperator::New();
  switch (activation->Kind()) {
    case MLOperator::OpKind::kClamp: {
      const MLClampOptions* ml_options =
          static_cast<const MLClampOptions*>(activation->Options());
      remote_graph->AddFusionClamp(BlinkClampOptioinToMojo(ml_options),
                                   activation->GetObjectId());

      fusion_operation->fusion_type = FusionType::kClamp;
      break;
    }
    case MLOperator::OpKind::kRelu: {
      fusion_operation->fusion_type = FusionType::kRelu;
      break;
    }
    default: {
      NOTREACHED();
      break;
    }
  }
  fusion_operation->object_id = activation->GetObjectId();
  return fusion_operation;
}

void AddConv2d(const MLOperator* conv2d,
               OperandDescriptorPtr output_desc,
               ml::webnn::mojom::blink::Graph* remote_graph) {
  const MLConv2dOptions* ml_options =
      static_cast<const MLConv2dOptions*>(conv2d->Options());
  auto* input = conv2d->Inputs()[0].Get();
  auto options = ml::webnn::mojom::blink::Conv2dOptions::New();
  options->padding =
      ml_options->hasPadding() ? ml_options->padding() : Vector<int32_t>(4, 0);
  options->strides =
      ml_options->hasStrides() ? ml_options->strides() : Vector<int32_t>(2, 1);
  options->dilations = ml_options->hasDilations() ? ml_options->dilations()
                                                  : Vector<int32_t>(2, 1);
  options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
  options->groups = ml_options->groups();
  options->inputLayout =
      BlinkInputOperandLayoutToMojo(ml_options->inputLayout().AsEnum());
  options->filterLayout =
      BlinkConv2dFilterOperandLayoutToMojo(ml_options->filterLayout().AsEnum());
  options->bias_id =
      ml_options->hasBias() ? ml_options->bias()->GetObjectId() : 0;
  if (ml_options->hasActivation()) {
    options->activation =
        AddFusionOperator(ml_options->activation(), remote_graph);
  }

  auto* filter = conv2d->Inputs()[1].Get();
  remote_graph->AddConv2d(input->GetObjectId(), filter->GetObjectId(),
                          std::move(options), std::move(output_desc));
}

void AddPool2d(const MLOperator* pool2d,
               OperandDescriptorPtr output_desc,
               ml::webnn::mojom::blink::Graph* remote_graph) {
  const MLPool2dOptions* ml_options =
      static_cast<const MLPool2dOptions*>(pool2d->Options());
  auto* input = pool2d->Inputs()[0].Get();

  auto options = ml::webnn::mojom::blink::Pool2dOptions::New();
  options->window_dimensions = ml_options->hasWindowDimensions()
                                   ? ml_options->windowDimensions()
                                   : Vector<int32_t>();
  options->padding =
      ml_options->hasPadding() ? ml_options->padding() : Vector<int32_t>(4, 0);
  options->strides =
      ml_options->hasStrides() ? ml_options->strides() : Vector<int32_t>(2, 1);
  options->dilations = ml_options->hasDilations() ? ml_options->dilations()
                                                  : Vector<int32_t>(2, 1);
  options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
  options->layout =
      BlinkInputOperandLayoutToMojo(ml_options->layout().AsEnum());
  options->rounding_type =
      BlinkRoundingTypeToMojo(ml_options->roundingType().AsEnum());
  options->output_sizes = ml_options->hasOutputSizes()
                              ? ml_options->outputSizes()
                              : Vector<int32_t>();
  remote_graph->AddPool2d(input->GetObjectId(), std::move(options),
                          BlinkPool2dTypeToMojo(pool2d->Kind()),
                          std::move(output_desc));
}

void AddGemm(const MLOperator* gemm,
             OperandDescriptorPtr output_desc,
             ml::webnn::mojom::blink::Graph* remote_graph) {
  const MLGemmOptions* ml_options =
      static_cast<const MLGemmOptions*>(gemm->Options());
  auto* input_a = gemm->Inputs()[0].Get();
  auto* input_b = gemm->Inputs()[1].Get();
  auto options = ml::webnn::mojom::blink::GemmOptions::New();
  options->c_id = ml_options->hasC() ? ml_options->c()->GetObjectId() : 0;
  options->alpha = ml_options->hasAlpha() ? ml_options->alpha() : 1.0;
  options->beta = ml_options->hasBeta() ? ml_options->beta() : 1.0;
  options->a_transpose =
      ml_options->hasATranspose() ? ml_options->aTranspose() : false;
  options->b_transpose =
      ml_options->hasBTranspose() ? ml_options->bTranspose() : false;
  remote_graph->AddGemm(input_a->GetObjectId(), input_b->GetObjectId(),
                        std::move(options), std::move(output_desc));
}

void AddBinary(const MLOperator* binary,
               OperandDescriptorPtr output_desc,
               ml::webnn::mojom::blink::Graph* remote_graph) {
  auto* input0 = binary->Inputs()[0].Get();
  auto* input1 = binary->Inputs()[1].Get();
  remote_graph->AddElementWiseBinary(
      input0->GetObjectId(), input1->GetObjectId(),
      BlinkBinaryOperandTypeToMojo(binary->Kind()), std::move(output_desc));
}

void AddUnary(const MLOperator* unary,
              OperandDescriptorPtr output_desc,
              ml::webnn::mojom::blink::Graph* remote_graph) {
  auto* input = unary->Inputs()[0].Get();
  remote_graph->AddUnary(input->GetObjectId(),
                         BlinkUnaryOperandTypeToMojo(unary->Kind()),
                         std::move(output_desc));
}

void AddReshape(const MLOperator* reshape,
                OperandDescriptorPtr output_desc,
                ml::webnn::mojom::blink::Graph* remote_graph) {
  auto* input = reshape->Inputs()[0].Get();
  remote_graph->AddReshape(input->GetObjectId(), std::move(output_desc));
}

typedef void (*OperatorFunc)(const MLOperator*,
                             OperandDescriptorPtr,
                             ml::webnn::mojom::blink::Graph*);

OperatorFunc GetOperatorFunc(MLOperator::OpKind type) {
  switch (type) {
    case MLOperator::OpKind::kClamp:
      return AddClamp;
    case MLOperator::OpKind::kConv2d:
      return AddConv2d;
    case MLOperator::OpKind::kAdd:
      return AddBinary;
    case MLOperator::OpKind::kGemm:
      return AddGemm;
    case MLOperator::OpKind::kAveragePool2d:
      return AddPool2d;
    case MLOperator::OpKind::kReshape:
      return AddReshape;
    case MLOperator::OpKind::kRelu:
    case MLOperator::OpKind::kSoftmax:
      return AddUnary;
    default:
      NOTREACHED();
      return nullptr;
  }
}

}  // namespace

MojoGraph::MojoGraph(ScriptState* script_state, MLContext* context)
    : MLGraph(context), remote_graph_(ExecutionContext::From(script_state)) {}

MojoGraph::~MojoGraph() = default;

ScriptPromise MojoGraph::BuildImpl(ScriptState* script_state,
                                   MLNamedOperands named_outputs,
                                   ExceptionState& exception_state) {
  auto* request = MakeGarbageCollected<BuildRequest>(std::move(named_outputs));
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  MojoContext* mojo_context = static_cast<MojoContext*>(GetContext());
  mojo_context->CreateGraph(
      resolver, GetObjectId(),
      WTF::Bind(&MojoGraph::OnGraphCreated, WrapPersistent(this),
                WrapPersistent(script_state), WrapPersistent(resolver),
                WrapPersistent(request)));

  return resolver->Promise();
}

void MojoGraph::BuildSyncImpl(MLNamedOperands named_outputs,
                              ExceptionState& exception_state) {
  NOTIMPLEMENTED();
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented");
}

ScriptPromise MojoGraph::ComputeImpl(ScriptState* script_state,
                                     MLNamedArrayInputs inputs,
                                     MLNamedArrayOutputs outputs,
                                     ExceptionState& exception_state) {
  if (inputs.size() != inputs_byte_length_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The number of inputs is invalid");
    return ScriptPromise();
  }
  auto named_inputs = ml::webnn::mojom::blink::NamedInputs::New();
  for (const auto& input : inputs) {
    String error_message;
    DOMArrayBufferView* array_buffer_view =
        ValidateInputBuffer(input, error_message);
    if (array_buffer_view == nullptr) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        error_message);
      return ScriptPromise();
    }
    const String& input_name = input.first;
    auto memory_info = ml::webnn::mojom::blink::MemoryInfo::New();
    memory_info->byte_offset = inputs_byte_offset_.at(input_name);
    memory_info->byte_length = inputs_byte_length_.at(input_name);
    uint8_t* address = inputs_shm_region_.mapping.GetMemoryAs<uint8_t>() +
                       memory_info->byte_offset;
    memcpy(address, array_buffer_view->BaseAddressMaybeShared(),
           array_buffer_view->byteLength());
    named_inputs->inputs.insert(input_name, std::move(memory_info));
  }
  named_inputs->shared_memory = inputs_shm_region_.region.Duplicate();

  ScriptPromiseResolver* resolver =
      MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  auto* request = MakeGarbageCollected<ComputeRequest>(std::move(inputs),
                                                       std::move(outputs));
  remote_graph_->Compute(
      std::move(named_inputs),
      WTF::Bind(&MojoGraph::OnGraphComputed, WrapPersistent(this),
                WrapPersistent(resolver), WrapPersistent(request)));
  return resolver->Promise();
}

void MojoGraph::ComputeSyncImpl(MLNamedArrayInputs inputs,
                                MLNamedArrayOutputs outputs,
                                ExceptionState& exception_state) {
  NOTIMPLEMENTED();
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented");
}

void MojoGraph::Trace(Visitor* visitor) const {
  visitor->Trace(remote_graph_);
  MLGraph::Trace(visitor);
}

void MojoGraph::OnGraphCreated(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    BuildRequest* request,
    mojo::PendingRemote<ml::webnn::mojom::blink::Graph> pending_remote) {
  auto* execution_context = ExecutionContext::From(script_state);
  remote_graph_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  base::CheckedNumeric<size_t> inputs_buffer_length(0);
  for (const auto& input : request->inputs_) {
    size_t input_byte_length = 0;
    if (!input->CalculateByteLength(input_byte_length)) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "Overflow occurred when calcuating byte length of input: " +
              input->Name()));
      return;
    }
    input_byte_length = Align(input_byte_length, kBufferAlignment).ValueOrDie();
    auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    desc->data_type = BlinkOperandTypeToMojo(input->Type());
    desc->dimensions = input->Dimensions();
    desc->object_id = input->GetObjectId();
    remote_graph_->AddInput(input->Name(), std::move(desc));

    inputs_byte_length_.insert(input->Name(), input_byte_length);
    inputs_byte_offset_.insert(input->Name(),
                               inputs_buffer_length.ValueOrDie());
    inputs_buffer_length += input_byte_length;
  }
  inputs_shm_region_ = base::ReadOnlySharedMemoryRegion::Create(
      inputs_buffer_length.ValueOrDie());

  base::CheckedNumeric<size_t> constants_buffer_length(0);
  for (const auto& constant : request->constants_) {
    wtf_size_t size = base::checked_cast<wtf_size_t>(
        constant->ArrayBufferView()->byteLength());
    constants_buffer_length += Align(size, kBufferAlignment);
  }
  base::MappedReadOnlyRegion constants_shm_region =
      base::ReadOnlySharedMemoryRegion::Create(
          constants_buffer_length.ValueOrDie());
  auto constants_info = ml::webnn::mojom::blink::ConstantsInfo::New();
  base::CheckedNumeric<size_t> constant_offset(0);
  for (const auto& constant : request->constants_) {
    auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    desc->data_type = BlinkOperandTypeToMojo(constant->Type());
    desc->dimensions = constant->Dimensions();
    desc->object_id = constant->GetObjectId();

    auto* array_buffer_view = constant->ArrayBufferView();
    size_t size =
        Align(array_buffer_view->byteLength(), kBufferAlignment).ValueOrDie();
    remote_graph_->AddConstant(std::move(desc));

    auto memory_info = ml::webnn::mojom::blink::MemoryInfo::New();
    memory_info->byte_offset = constant_offset.ValueOrDie();
    memory_info->byte_length = size;
    constant_offset += size;

    uint8_t* address = constants_shm_region.mapping.GetMemoryAs<uint8_t>() +
                       memory_info->byte_offset;
    memcpy(address, array_buffer_view->BaseAddressMaybeShared(),
           array_buffer_view->byteLength());
    constants_info->constants.insert(constant->GetObjectId(),
                                     std::move(memory_info));
  }
  constants_info->shared_memory = constants_shm_region.region.Duplicate();

  for (const auto& op : request->sorted_operators_) {
    auto* output = op->Outputs()[0].Get();
    auto output_desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    output_desc->dimensions = output->Dimensions();
    output_desc->object_id = output->GetObjectId();
    OperatorFunc add_operator_func = GetOperatorFunc(op->Kind());
    if (add_operator_func == nullptr) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "Failed to get function to add the operator"));
      return;
    }
    add_operator_func(op, std::move(output_desc), remote_graph_.get());
  }

  WTF::HashMap<WTF::String, uint32_t> named_operands;
  for (const auto& [name, output] : request->outputs_) {
    size_t output_byte_length = 0;
    if (!output->CalculateByteLength(output_byte_length)) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "Overflow occurred when calcuating byte length of output: " + name));
      return;
    }

    outputs_byte_length_.insert(name, output_byte_length);
    named_operands.insert(name, output->GetObjectId());
    // remote_graph_->AddOutput(name, output->GetObjectId());
  }
  remote_graph_->Build(
      std::move(named_operands),
      request->constants_.size() == 0 ? nullptr : std::move(constants_info),
      WTF::Bind(&MojoGraph::OnGraphBuilt, WrapPersistent(this),
                WrapPersistent(resolver)));
  return;
}

void MojoGraph::OnGraphBuilt(ScriptPromiseResolver* resolver,
                             BuildResult result) {
  switch (result) {
    case BuildResult::kUnknownError: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Internal error."));
      return;
    }
    case BuildResult::kOk: {
      resolver->Resolve(this);
      return;
    }
  }
}

void MojoGraph::OnGraphComputed(ScriptPromiseResolver* resolver,
                                ComputeRequest* request,
                                ComputeResult result,
                                NamedOutputsPtr named_outputs) {
  if (result != ComputeResult::kOk) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kOperationError,
        "Failed to obtain the computation result."));
    return;
  }
  for (const auto& output : request->outputs_) {
    String error_message;
    void* output_buffer_address = ValidateOutputBuffer(output, error_message);
    if (output_buffer_address == nullptr) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError, error_message));
      return;
    }
    auto iter = named_outputs->outputs.find(output.first);
    if (iter == named_outputs->outputs.end()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError,
          "Failed to get result for the output."));
      return;
    }
    MemoryInfoPtr memory_info = std::move(iter->value);
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        named_outputs->shared_memory;
    DCHECK(shared_memory_region.IsValid());
    size_t byte_length = base::checked_cast<size_t>(memory_info->byte_length);
    base::ReadOnlySharedMemoryMapping shared_memory_mapping =
        shared_memory_region.MapAt(memory_info->byte_offset, byte_length);
    memcpy(output_buffer_address, shared_memory_mapping.GetMemoryAs<uint8_t>(),
           byte_length);
  }
  resolver->Resolve();
  return;
}

}  // namespace blink

// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_buffer_resource_view.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_input.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_leaky_relu_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_type.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_arraybufferview_mlbufferresourceview.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_arraybufferview_mlbufferresourceview_mlinput.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

namespace blink {

WNNOperandDescriptor AsWebnnType(const MLOperandDescriptor* desc) {
  WNNOperandDescriptor webnn_desc;
  switch (desc->type().AsEnum()) {
    case V8MLOperandType::Enum::kFloat32:
      webnn_desc.type = WNNOperandType_Float32;
      break;
    case V8MLOperandType::Enum::kFloat16:
      webnn_desc.type = WNNOperandType_Float16;
      break;
    case V8MLOperandType::Enum::kInt32:
      webnn_desc.type = WNNOperandType_Int32;
      break;
    case V8MLOperandType::Enum::kUint32:
      webnn_desc.type = WNNOperandType_Uint32;
      break;
    case V8MLOperandType::Enum::kInt8:
      webnn_desc.type = WNNOperandType_Int8;
      break;
    case V8MLOperandType::Enum::kUint8:
      webnn_desc.type = WNNOperandType_Uint8;
      break;
  }
  webnn_desc.dimensions = desc->dimensions().data();
  webnn_desc.dimensionsCount = desc->dimensions().size();
  return webnn_desc;
}

WNNAutoPad AsWebnnType(const V8MLAutoPad auto_pad) {
  WNNAutoPad webnn_auto_pad;
  switch (auto_pad.AsEnum()) {
    case V8MLAutoPad::Enum::kExplicit:
      webnn_auto_pad = WNNAutoPad_Explicit;
      break;
    case V8MLAutoPad::Enum::kSameUpper:
      webnn_auto_pad = WNNAutoPad_SameUpper;
      break;
    case V8MLAutoPad::Enum::kSameLower:
      webnn_auto_pad = WNNAutoPad_SameLower;
      break;
  }
  return webnn_auto_pad;
}

WNNInputOperandLayout AsWebnnType(const V8MLInputOperandLayout input_layout) {
  WNNInputOperandLayout webnn_input_layout;
  switch (input_layout.AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      webnn_input_layout = WNNInputOperandLayout_Nchw;
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      webnn_input_layout = WNNInputOperandLayout_Nhwc;
      break;
  }
  return webnn_input_layout;
}

WNNConv2dFilterOperandLayout AsWebnnType(
    const V8MLConv2dFilterOperandLayout filter_layout) {
  WNNConv2dFilterOperandLayout webnn_filter_layout;
  switch (filter_layout.AsEnum()) {
    case V8MLConv2dFilterOperandLayout::Enum::kOihw:
      webnn_filter_layout = WNNConv2dFilterOperandLayout_Oihw;
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kHwio:
      webnn_filter_layout = WNNConv2dFilterOperandLayout_Hwio;
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
      webnn_filter_layout = WNNConv2dFilterOperandLayout_Ohwi;
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
      webnn_filter_layout = WNNConv2dFilterOperandLayout_Ihwo;
      break;
  }
  return webnn_filter_layout;
}

WNNConv2dOptions AsWebnnType(const MLConv2dOptions* conv2d_options) {
  WNNConv2dOptions webnn_conv2d_options;
  webnn_conv2d_options.padding =
      conv2d_options->hasPadding() ? conv2d_options->padding().data() : nullptr;
  webnn_conv2d_options.paddingCount =
      conv2d_options->hasPadding() ? conv2d_options->padding().size() : 0;
  webnn_conv2d_options.strides =
      conv2d_options->hasStrides() ? conv2d_options->strides().data() : nullptr;
  webnn_conv2d_options.stridesCount =
      conv2d_options->hasStrides() ? conv2d_options->strides().size() : 0;
  webnn_conv2d_options.dilations = conv2d_options->hasDilations()
                                       ? conv2d_options->dilations().data()
                                       : nullptr;
  webnn_conv2d_options.dilationsCount =
      conv2d_options->hasDilations() ? conv2d_options->dilations().size() : 0;
  webnn_conv2d_options.autoPad = AsWebnnType(conv2d_options->autoPad());
  webnn_conv2d_options.groups = conv2d_options->groups();
  webnn_conv2d_options.inputLayout = AsWebnnType(conv2d_options->inputLayout());
  webnn_conv2d_options.filterLayout =
      AsWebnnType(conv2d_options->filterLayout());
  webnn_conv2d_options.bias =
      conv2d_options->hasBias() ? conv2d_options->bias()->GetHandle() : nullptr;
  webnn_conv2d_options.activation =
      conv2d_options->hasActivation()
          ? conv2d_options->activation()->GetHandle()
          : nullptr;
  return webnn_conv2d_options;
}

WNNGemmOptions AsWebnnType(const MLGemmOptions* gemm_options) {
  WNNGemmOptions webnn_gemm_options;
  webnn_gemm_options.c =
      gemm_options->hasC() ? gemm_options->c()->GetHandle() : nullptr;
  webnn_gemm_options.alpha = gemm_options->alpha();
  webnn_gemm_options.beta = gemm_options->beta();
  webnn_gemm_options.aTranspose = gemm_options->aTranspose();
  webnn_gemm_options.bTranspose = gemm_options->bTranspose();
  return webnn_gemm_options;
}

WNNPool2dOptions AsWebnnType(const MLPool2dOptions* pool2d_options) {
  WNNPool2dOptions webnn_pool2d_options;
  webnn_pool2d_options.windowDimensions =
      pool2d_options->hasWindowDimensions()
          ? pool2d_options->windowDimensions().data()
          : nullptr;
  webnn_pool2d_options.windowDimensionsCount =
      pool2d_options->hasWindowDimensions()
          ? pool2d_options->windowDimensions().size()
          : 0;
  webnn_pool2d_options.padding =
      pool2d_options->hasPadding() ? pool2d_options->padding().data() : nullptr;
  webnn_pool2d_options.paddingCount =
      pool2d_options->hasPadding() ? pool2d_options->padding().size() : 0;
  webnn_pool2d_options.strides =
      pool2d_options->hasStrides() ? pool2d_options->strides().data() : nullptr;
  webnn_pool2d_options.stridesCount =
      pool2d_options->hasStrides() ? pool2d_options->strides().size() : 0;
  webnn_pool2d_options.dilations = pool2d_options->hasDilations()
                                       ? pool2d_options->dilations().data()
                                       : nullptr;
  webnn_pool2d_options.dilationsCount =
      pool2d_options->hasDilations() ? pool2d_options->dilations().size() : 0;
  webnn_pool2d_options.autoPad = AsWebnnType(pool2d_options->autoPad());
  webnn_pool2d_options.layout = AsWebnnType(pool2d_options->layout());
  return webnn_pool2d_options;
}

WNNArrayBufferView AsWebnnType(const MLBufferResourceView* resource) {
  NOTREACHED();
  WNNArrayBufferView webnn_buffer;
  //   webnn_buffer.buffer = resource->resource()->GetHandle();
  //   webnn_buffer.byteLength = resource->getSizeOr(0);
  //   webnn_buffer.byteOffset = resource->offset();
  return webnn_buffer;
}

WNNArrayBufferView AsWebnnType(const NotShared<DOMArrayBufferView>& resource) {
  WNNArrayBufferView webnn_buffer;
  webnn_buffer.buffer = resource->BaseAddress();
  webnn_buffer.byteLength = resource->byteLength();
  webnn_buffer.byteOffset = resource->byteOffset();
  return webnn_buffer;
}

WNNArrayBufferView AsWebnnType(const MLResource* buffer) {
  DCHECK(buffer);
  switch (buffer->GetContentType()) {
    case MLResource::ContentType::kArrayBufferView:
      return AsWebnnType(buffer->GetAsArrayBufferView());
    case MLResource::ContentType::kMLBufferResourceView:
      NOTREACHED();
      return {};
  }
  NOTREACHED();
  return {};
}

WNNInput AsWebnnType(const MLInputResource* buffer) {
  DCHECK(buffer);
  WNNInput webnn_input = {};
  switch (buffer->GetContentType()) {
    case MLInputResource::ContentType::kArrayBufferView:
      webnn_input.resource = AsWebnnType(buffer->GetAsArrayBufferView());
      break;
    case MLInputResource::ContentType::kMLBufferResourceView:
      NOTREACHED();
      break;
    case MLInputResource::ContentType::kMLInput: {
      MLInput* ml_input = buffer->GetAsMLInput();
      webnn_input.resource = AsWebnnType(ml_input->resource());
      webnn_input.dimensions =
          ml_input->hasDimensions() ? ml_input->dimensions().data() : nullptr;
      webnn_input.dimensionsCount =
          ml_input->hasDimensions() ? ml_input->dimensions().size() : 0;
      break;
    }
    default:
      NOTREACHED();
  }
  return webnn_input;
}

// static
MLGraphBuilder* MLGraphBuilder::Create(MLContext* context) {
  // Get WebNNInstance
  WNNInstance instance = context->GetML()->GetInstance();
  WNNGraphBuilder webnn_builder =
      context->GetProcs().instanceCreateGraphBuilder(instance,
                                                     context->GetHandle());
  MLGraphBuilder* builder =
      MakeGarbageCollected<MLGraphBuilder>(context, webnn_builder);
  return builder;
}

MLGraphBuilder::MLGraphBuilder(MLContext* context, WNNGraphBuilder builder)
    : WebnnObject<WNNGraphBuilder>(context, builder) {}

void MLGraphBuilder::Trace(Visitor* visitor) const {
  WebnnObject<WNNGraphBuilder>::Trace(visitor);
}

MLContext* MLGraphBuilder::GetContext() const {
  return context_.Get();
}

MLOperand* MLGraphBuilder::input(String name, const MLOperandDescriptor* desc) {
  WNNOperandDescriptor webnn_desc = AsWebnnType(desc);
  std::string name_str = name.Utf8();
  WNNOperand webnn_input =
      GetProcs().graphBuilderInput(GetHandle(), name_str.c_str(), &webnn_desc);
  MLOperand* input = MakeGarbageCollected<MLOperand>(GetContext(), webnn_input);
  return input;
}

MLOperand* MLGraphBuilder::constant(const MLOperandDescriptor* desc,
                                    const MLResource* buffer_view) {
  WNNOperandDescriptor webnn_desc = AsWebnnType(desc);
  WNNArrayBufferView webnn_buffer_view = AsWebnnType(buffer_view);
  WNNOperand webnn_constant = GetProcs().graphBuilderConstant(
      GetHandle(), &webnn_desc, &webnn_buffer_view);
  MLOperand* constant =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_constant);
  return constant;
}

MLOperand* MLGraphBuilder::add(const MLOperand* a, const MLOperand* b) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderAdd(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::clamp(const MLOperand* input,
                                 const MLClampOptions* options) {
  WNNClampOptions webnn_clamp_options;
  webnn_clamp_options.minValue =
      options->getMinValueOr(std::numeric_limits<float>::lowest());
  webnn_clamp_options.maxValue =
      options->getMaxValueOr(std::numeric_limits<float>::max());
  WNNOperand webnn_output = GetProcs().graphBuilderClamp(
      GetHandle(), input->GetHandle(), &webnn_clamp_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperator* MLGraphBuilder::clamp(const MLClampOptions* options) {
  WNNClampOptions webnn_clamp_options;
  webnn_clamp_options.minValue =
      options->getMinValueOr(std::numeric_limits<float>::lowest());
  webnn_clamp_options.maxValue =
      options->getMaxValueOr(std::numeric_limits<float>::max());
  WNNFusionOperator webnn_operator =
      GetProcs().graphBuilderClampOperator(GetHandle(), &webnn_clamp_options);
  MLOperator* ml_operator =
      MakeGarbageCollected<MLOperator>(GetContext(), webnn_operator);
  return ml_operator;
}

MLOperand* MLGraphBuilder::concat(const HeapVector<Member<MLOperand>>& inputs,
                                  int32_t axis) {
  std::vector<WNNOperand> webnn_operands;
  for (unsigned int i = 0; i < inputs.size(); ++i) {
    webnn_operands.push_back(inputs[i]->GetHandle());
  }
  WNNOperand webnn_output = GetProcs().graphBuilderConcat(
      GetHandle(), static_cast<uint32_t>(webnn_operands.size()),
      webnn_operands.data(), axis);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::conv2d(const MLOperand* input,
                                  const MLOperand* filter,
                                  const MLConv2dOptions* options) {
  WNNConv2dOptions webnn_conv2d_options = AsWebnnType(options);
  WNNOperand webnn_output =
      GetProcs().graphBuilderConv2d(GetHandle(), input->GetHandle(),
                                    filter->GetHandle(), &webnn_conv2d_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::gemm(const MLOperand* a,
                                const MLOperand* b,
                                const MLGemmOptions* options) {
  WNNGemmOptions webnn_gemm_options = AsWebnnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderGemm(
      GetHandle(), a->GetHandle(), b->GetHandle(), &webnn_gemm_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::leakyRelu(const MLOperand* input,
                                     const MLLeakyReluOptions* options) {
  WNNLeakyReluOptions webnn_leaky_relu_options;
  webnn_leaky_relu_options.alpha = options->alpha();
  WNNOperand webnn_output = GetProcs().graphBuilderLeakyRelu(
      GetHandle(), input->GetHandle(), &webnn_leaky_relu_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperator* MLGraphBuilder::leakyRelu(const MLLeakyReluOptions* options) {
  WNNLeakyReluOptions webnn_leaky_relu_options;
  webnn_leaky_relu_options.alpha = options->alpha();
  WNNFusionOperator webnn_operator = GetProcs().graphBuilderLeakyReluOperator(
      GetHandle(), &webnn_leaky_relu_options);
  MLOperator* ml_operator =
      MakeGarbageCollected<MLOperator>(GetContext(), webnn_operator);
  return ml_operator;
}

MLOperand* MLGraphBuilder::matmul(const MLOperand* a, const MLOperand* b) {
  WNNOperand webnn_output = GetProcs().graphBuilderMatmul(
      GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::averagePool2d(const MLOperand* input,
                                         const MLPool2dOptions* options) {
  WNNPool2dOptions webnn_pool2d_options = AsWebnnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderAveragePool2d(
      GetHandle(), input->GetHandle(), &webnn_pool2d_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::maxPool2d(const MLOperand* input,
                                     const MLPool2dOptions* options) {
  WNNPool2dOptions webnn_pool2d_options = AsWebnnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderMaxPool2d(
      GetHandle(), input->GetHandle(), &webnn_pool2d_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::relu(const MLOperand* input) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderRelu(GetHandle(), input->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperator* MLGraphBuilder::relu() {
  WNNFusionOperator webnn_operator =
      GetProcs().graphBuilderReluOperator(GetHandle());
  MLOperator* ml_operator =
      MakeGarbageCollected<MLOperator>(GetContext(), webnn_operator);
  return ml_operator;
}

MLOperand* MLGraphBuilder::reshape(const MLOperand* input,
                                   const Vector<int32_t>& new_shape) {
  WNNOperand webnn_output = GetProcs().graphBuilderReshape(
      GetHandle(), input->GetHandle(), new_shape.data(),
      static_cast<uint32_t>(new_shape.size()));
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::sigmoid(const MLOperand* input) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderSigmoid(GetHandle(), input->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperator* MLGraphBuilder::sigmoid() {
  WNNFusionOperator webnn_operator =
      GetProcs().graphBuilderSigmoidOperator(GetHandle());
  MLOperator* ml_operator =
      MakeGarbageCollected<MLOperator>(GetContext(), webnn_operator);
  return ml_operator;
}

MLOperand* MLGraphBuilder::softmax(const MLOperand* input) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderSoftmax(GetHandle(), input->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLGraph* MLGraphBuilder::build(const MLNamedOperands& outputs) {
  WNNInstance instance = context_.Get()->GetML()->GetInstance();
  WNNNamedOperands webnn_outputs =
      GetProcs().instanceCreateNamedOperands(instance);
  for (wtf_size_t i = 0; i < outputs.size(); ++i) {
    std::string name = outputs[i].first.Utf8();
    WNNOperand webnn_operand = outputs[i].second->GetHandle();
    GetProcs().namedOperandsSet(webnn_outputs, name.c_str(), webnn_operand);
  }
  WNNGraph webnn_graph =
      GetProcs().graphBuilderBuild(GetHandle(), webnn_outputs);
  MLGraph* graph = MakeGarbageCollected<MLGraph>(GetContext(), webnn_graph);
  return graph;
}

}  // namespace blink

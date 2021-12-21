// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_batch_normalization_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_input.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_leaky_relu_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_type.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pad_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_reduce_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_resample_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_transpose_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_arraybufferviewallowshared_mlbufferresourceview.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_arraybufferviewallowshared_mlbufferresourceview_mlinput.h"
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
  webnn_desc.dimensions =
      desc->hasDimensions() ? desc->dimensions().data() : nullptr;
  webnn_desc.dimensionsCount =
      desc->hasDimensions() ? desc->dimensions().size() : 0;
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
  WNNPool2dOptions webnn_pool2d_options = {};
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

WNNGpuBufferView AsWebnnType(const MLBufferResourceView* resource) {
  WNNGpuBufferView webnn_buffer = {};
  std::tuple<uint32_t, uint32_t> buffer = resource->resource()->GetBufferId();
  webnn_buffer.id = std::get<0>(buffer);
  webnn_buffer.generation = std::get<1>(buffer);
  webnn_buffer.size = resource->getSizeOr(0);
  webnn_buffer.offset = resource->offset();
  return webnn_buffer;
}

WNNArrayBufferView AsWebnnType(
    const MaybeShared<DOMArrayBufferView>& resource) {
  WNNArrayBufferView webnn_buffer;
  webnn_buffer.buffer = resource->BaseAddressMaybeShared();
  webnn_buffer.byteLength = resource->byteLength();
  webnn_buffer.byteOffset = resource->byteOffset();
  return webnn_buffer;
}

WNNInput AsWebnnType(const MLInputResource* buffer) {
  DCHECK(buffer);
  WNNInput webnn_input = {};
  switch (buffer->GetContentType()) {
    case MLInputResource::ContentType::kArrayBufferViewAllowShared:
      webnn_input.resource.arrayBufferView =
          AsWebnnType(buffer->GetAsArrayBufferViewAllowShared());
      break;
    case MLInputResource::ContentType::kMLBufferResourceView:
      webnn_input.resource.gpuBufferView =
          AsWebnnType(buffer->GetAsMLBufferResourceView());
      break;
    case MLInputResource::ContentType::kMLInput: {
      MLInput* ml_input = buffer->GetAsMLInput();
      MLResource* resource = ml_input->resource();
      switch (resource->GetContentType()) {
        case MLResource::ContentType::kArrayBufferViewAllowShared:
          webnn_input.resource.arrayBufferView =
              AsWebnnType(resource->GetAsArrayBufferViewAllowShared());
          break;
        case MLResource::ContentType::kMLBufferResourceView:
          webnn_input.resource.gpuBufferView =
              AsWebnnType(resource->GetAsMLBufferResourceView());
          break;
        default:
          NOTREACHED();
      }
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

WNNBatchNormOptions AsWebnnType(
    const MLBatchNormalizationOptions* batch_norm_options) {
  WNNBatchNormOptions webnn_batch_norm_options;
  webnn_batch_norm_options.scale =
      batch_norm_options->hasScale() ? batch_norm_options->scale()->GetHandle()
                                     : nullptr;
  webnn_batch_norm_options.bias = batch_norm_options->hasBias()
                                      ? batch_norm_options->bias()->GetHandle()
                                      : nullptr;
  webnn_batch_norm_options.axis = batch_norm_options->axis();
  webnn_batch_norm_options.epsilon = batch_norm_options->epsilon();
  webnn_batch_norm_options.activation =
      batch_norm_options->hasActivation()
          ? batch_norm_options->activation()->GetHandle()
          : nullptr;
  return webnn_batch_norm_options;
}

WNNPadOptions AsWebnnType(const MLPadOptions* pad_options) {
  WNNPadOptions webnn_pad_options;
  switch (pad_options->mode().AsEnum()) {
    case V8MLPaddingMode::Enum::kConstant:
      webnn_pad_options.mode = WNNPaddingMode_Constant;
      break;
    case V8MLPaddingMode::Enum::kEdge:
      webnn_pad_options.mode = WNNPaddingMode_Edge;
      break;
    case V8MLPaddingMode::Enum::kReflection:
      webnn_pad_options.mode = WNNPaddingMode_Reflection;
      break;
    case V8MLPaddingMode::Enum::kSymmetric:
      webnn_pad_options.mode = WNNPaddingMode_Symmetric;
      break;
  }
  webnn_pad_options.value = pad_options->value();
  return webnn_pad_options;
}

WNNResample2dOptions AsDawnType(const MLResample2dOptions* options) {
  WNNResample2dOptions webnn_options;
  switch (options->mode().AsEnum()) {
    case V8MLInterpolationMode::Enum::kNearestNeighbor:
      webnn_options.mode = WNNInterpolationMode_NearestNeighbor;
      break;
    case V8MLInterpolationMode::Enum::kLinear:
      webnn_options.mode = WNNInterpolationMode_Linear;
      break;
  }
  webnn_options.scalesCount =
      options->hasScales() ? options->scales().size() : 0;
  webnn_options.scales =
      options->hasScales() ? options->scales().data() : nullptr;
  webnn_options.sizesCount = options->hasSizes() ? options->sizes().size() : 0;
  webnn_options.sizes = options->hasSizes() ? options->sizes().data() : nullptr;
  webnn_options.axesCount = options->hasAxes() ? options->axes().size() : 0;
  webnn_options.axes = options->hasAxes() ? options->axes().data() : nullptr;
  return webnn_options;
}

WNNReduceOptions AsDawnType(const MLReduceOptions* options) {
  WNNReduceOptions webnn_options;
  webnn_options.axesCount = options->hasAxes() ? options->axes().size() : 0;
  webnn_options.axes = options->hasAxes() ? options->axes().data() : nullptr;
  webnn_options.keepDimensions = options->keepDimensions();
  return webnn_options;
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
  visitor->Trace(gpu_buffers_);
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
  WNNOperand webnn_constant;
  switch (buffer_view->GetContentType()) {
    case MLResource::ContentType::kArrayBufferViewAllowShared: {
      WNNArrayBufferView array_buffer_view =
          AsWebnnType(buffer_view->GetAsArrayBufferViewAllowShared());
      webnn_constant = GetProcs().graphBuilderConstant(GetHandle(), &webnn_desc,
                                                       &array_buffer_view);
      break;
    }
    case MLResource::ContentType::kMLBufferResourceView: {
      // Reference the GPUBuffer.
      gpu_buffers_.push_back(
          buffer_view->GetAsMLBufferResourceView()->resource());
      WNNGpuBufferView gpu_buffer_view =
          AsWebnnType(buffer_view->GetAsMLBufferResourceView());
      webnn_constant = GetProcs().graphBuilderConstantWithGpuBuffer(
          GetHandle(), &webnn_desc, &gpu_buffer_view);
      break;
    }
    default:
      NOTREACHED();
  }
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

MLOperand* MLGraphBuilder::sub(const MLOperand* a, const MLOperand* b) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderSub(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::mul(const MLOperand* a, const MLOperand* b) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderMul(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::div(const MLOperand* a, const MLOperand* b) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderDiv(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::max(const MLOperand* a, const MLOperand* b) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderMax(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::min(const MLOperand* a, const MLOperand* b) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderMin(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::pow(const MLOperand* a, const MLOperand* b) {
  WNNOperand webnn_output =
      GetProcs().graphBuilderPow(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::batchNormalization(
    const MLOperand* input,
    const MLOperand* mean,
    const MLOperand* variance,
    const MLBatchNormalizationOptions* options) {
  WNNBatchNormOptions webnn_batch_norm_options = AsWebnnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderBatchNorm(
      GetHandle(), input->GetHandle(), mean->GetHandle(), variance->GetHandle(),
      &webnn_batch_norm_options);
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

MLOperand* MLGraphBuilder::pad(const MLOperand* input,
                               const Vector<uint32_t>& padding,
                               const MLPadOptions* options) {
  // WNNPadOptions webnn_pad_options = AsWebnnType(options);
  // WNNOperand webnn_output = GetProcs().graphBuilderPad(
  //     GetHandle(), input->GetHandle(), padding.data(),
  //     static_cast<uint32_t>(padding.size()), &webnn_pad_options);
  // MLOperand* output =
  //     MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  // return output;
  return nullptr;
}

MLOperand* MLGraphBuilder::reduceArgMax(const MLOperand* input,
                                        const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceArgMax(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::reduceArgMin(const MLOperand* input,
                                        const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceArgMin(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::reduceL1(const MLOperand* input,
                                    const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceL1(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::reduceL2(const MLOperand* input,
                                    const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceL2(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::reduceMax(const MLOperand* input,
                                     const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceMax(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::reduceMean(const MLOperand* input,
                                      const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceMean(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::reduceMin(const MLOperand* input,
                                     const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceMin(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::reduceProduct(const MLOperand* input,
                                         const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceProduct(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
}

MLOperand* MLGraphBuilder::reduceSum(const MLOperand* input,
                                     const MLReduceOptions* options) {
  WNNReduceOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderReduceSum(
      GetHandle(), input->GetHandle(), &webnn_options);
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

MLOperand* MLGraphBuilder::resample2d(const MLOperand* input,
                                      const MLResample2dOptions* options) {
  WNNResample2dOptions webnn_options = AsDawnType(options);
  WNNOperand webnn_output = GetProcs().graphBuilderResample2d(
      GetHandle(), input->GetHandle(), &webnn_options);
  MLOperand* output =
      MakeGarbageCollected<MLOperand>(GetContext(), webnn_output);
  return output;
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

MLOperand* MLGraphBuilder::transpose(const MLOperand* input,
                                     const MLTransposeOptions* options) {
  WNNTransposeOptions webnn_transpose_options;
  webnn_transpose_options.permutationCount =
      options->hasPermutation() ? options->permutation().size() : 0;
  webnn_transpose_options.permutation =
      options->hasPermutation() ? options->permutation().data() : nullptr;
  WNNOperand webnn_output = GetProcs().graphBuilderTranspose(
      GetHandle(), input->GetHandle(), &webnn_transpose_options);
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

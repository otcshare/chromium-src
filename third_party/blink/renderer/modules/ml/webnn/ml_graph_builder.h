// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_buffer_resource_view.h"
#include "third_party/blink/renderer/core/typed_arrays/array_buffer_view_helpers.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_buffer.h"

namespace blink {

class MLContext;
class MLClampOptions;
class MLContext;
class MLConv2dOptions;
class MLGemmOptions;
class MLLeakyReluOptions;
class MLBatchNormalizationOptions;
class MLPadOptions;
class MLPool2dOptions;
class MLOperand;
class MLOperandDescriptor;
class MLOperator;
class MLGraph;
class MLReduceOptions;
class MLResample2dOptions;
class MLTransposeOptions;
class V8UnionArrayBufferViewAllowSharedOrMLBufferResourceViewOrMLInput;
class V8UnionArrayBufferViewAllowSharedOrMLBufferResourceView;

typedef V8UnionArrayBufferViewAllowSharedOrMLBufferResourceViewOrMLInput
    MLInputResource;
typedef V8UnionArrayBufferViewAllowSharedOrMLBufferResourceView MLResource;

WNNInput AsWebnnType(const MLInputResource* input);
WNNArrayBufferView AsWebnnType(const MLResource* resource);
WNNArrayBufferView AsWebnnType(const MaybeShared<DOMArrayBufferView>& resource);
WNNOperandDescriptor AsWebnnType(const MLOperandDescriptor* desc);
WNNGpuBufferView AsWebnnType(const MLBufferResourceView* resource);

typedef HeapVector<std::pair<String, Member<MLOperand>>> MLNamedOperands;

class MLGraphBuilder : public WebnnObject<WNNGraphBuilder> {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static MLGraphBuilder* Create(MLContext* context);

  explicit MLGraphBuilder(MLContext* context, WNNGraphBuilder builder);

  MLGraphBuilder(const MLGraphBuilder&) = delete;
  MLGraphBuilder& operator=(const MLGraphBuilder&) = delete;

  void Trace(Visitor* visitor) const override;

  MLContext* GetContext() const;

  // ml_graph_builder.idl
  MLOperand* input(String name, const MLOperandDescriptor* desc);

  MLOperand* constant(const MLOperandDescriptor* desc,
                      const MLResource* buffer_view);

  MLOperand* add(const MLOperand* a, const MLOperand* b);
  MLOperand* sub(const MLOperand* a, const MLOperand* b);
  MLOperand* mul(const MLOperand* a, const MLOperand* b);
  MLOperand* div(const MLOperand* a, const MLOperand* b);
  MLOperand* max(const MLOperand* a, const MLOperand* b);
  MLOperand* min(const MLOperand* a, const MLOperand* b);
  MLOperand* pow(const MLOperand* a, const MLOperand* b);

  MLOperand* batchNormalization(const MLOperand* input,
                                const MLOperand* mean,
                                const MLOperand* variance,
                                const MLBatchNormalizationOptions* options);

  MLOperand* clamp(const MLOperand* input, const MLClampOptions* options);

  MLOperator* clamp(const MLClampOptions* options);

  MLOperand* concat(const HeapVector<Member<MLOperand>>& inputs, int32_t axis);

  MLOperand* conv2d(const MLOperand* input,
                    const MLOperand* filter,
                    const MLConv2dOptions* options);

  MLOperand* gemm(const MLOperand* a,
                  const MLOperand* b,
                  const MLGemmOptions* options);

  MLOperand* leakyRelu(const MLOperand* input,
                       const MLLeakyReluOptions* options);

  MLOperator* leakyRelu(const MLLeakyReluOptions* options);

  MLOperand* matmul(const MLOperand* a, const MLOperand* b);

  MLOperand* averagePool2d(const MLOperand* input,
                           const MLPool2dOptions* options);
  MLOperand* maxPool2d(const MLOperand* input, const MLPool2dOptions* options);

  MLOperand* pad(const MLOperand* input,
                 const Vector<uint32_t>& padding,
                 const MLPadOptions* options);

  MLOperand* reduceArgMax(const MLOperand* input, const MLReduceOptions* options);
  MLOperand* reduceArgMin(const MLOperand* input, const MLReduceOptions* options);
  MLOperand* reduceL1(const MLOperand* input, const MLReduceOptions* options);
  MLOperand* reduceL2(const MLOperand* input, const MLReduceOptions* options);
  MLOperand* reduceMax(const MLOperand* input, const MLReduceOptions* options);
  MLOperand* reduceMean(const MLOperand* input, const MLReduceOptions* options);
  MLOperand* reduceMin(const MLOperand* input, const MLReduceOptions* options);
  MLOperand* reduceProduct(const MLOperand* input, const MLReduceOptions* options);
  MLOperand* reduceSum(const MLOperand* input, const MLReduceOptions* options);

  MLOperand* relu(const MLOperand* input);
  MLOperator* relu();

  MLOperand* resample2d(const MLOperand* input, const MLResample2dOptions* options);

  MLOperand* reshape(const MLOperand* input, const Vector<int32_t>& new_shape);

  MLOperand* sigmoid(const MLOperand* input);

  MLOperator* sigmoid();

  MLOperand* softmax(const MLOperand* input);

  MLOperand* transpose(const MLOperand* input, const MLTransposeOptions* options);

  MLGraph* build(const MLNamedOperands& outputs);

 private:
  HeapVector<Member<GPUBuffer>> gpu_buffers_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_
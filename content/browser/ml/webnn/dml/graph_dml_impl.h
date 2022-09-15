// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_GRAPH_DML_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_GRAPH_DML_IMPL_H_

#define DML_TARGET_VERSION_USE_LATEST 1

#include <dxgi1_4.h>
#include <dxgi1_6.h>
#include <wrl\client.h>
#include <unordered_map>
#include <unordered_set>

#include <algorithm>
#include "DirectML.h"
#include "base/memory/read_only_shared_memory_region.h"
#include "base/memory/shared_memory_mapping.h"
#include "components/ml/mojom/webnn_graph.mojom.h"
#include "content/browser/ml/webnn/dml/gpgmm_d3d12.h"
#include "content/browser/ml/webnn/dml/graph_desc_builder.h"
#include "content/browser/ml/webnn/dml/graph_node_output.h"
#include "content/browser/ml/webnn/dml/readback_heap.h"
#include "content/browser/ml/webnn/dml/upload_heap.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "utils_dml.h"

namespace content::webnn {

namespace {

using namespace Microsoft::WRL;
using ml::webnn::mojom::BinaryOperandType;
using ml::webnn::mojom::BuildResult;
using ml::webnn::mojom::ClampOptions;
using ml::webnn::mojom::ClampOptionsPtr;
using ml::webnn::mojom::ComputeResult;
using ml::webnn::mojom::ConstantsInfoPtr;
using ml::webnn::mojom::Conv2dOptionsPtr;
using ml::webnn::mojom::FusionOperator;
using ml::webnn::mojom::GemmOptionsPtr;
using ml::webnn::mojom::NamedInputsPtr;
using ml::webnn::mojom::NamedOutputsPtr;
using ml::webnn::mojom::OperandDescriptorPtr;
using ml::webnn::mojom::Pool2dOptions;
using ml::webnn::mojom::Pool2dOptionsPtr;
using ml::webnn::mojom::Pool2dType;
using ml::webnn::mojom::UnaryOperandType;

}  // namespace

class FusionOperators;
class ExecutionContext;

class GraphDMLImpl : public ml::webnn::mojom::Graph {
 public:
  ~GraphDMLImpl() override;
  static void Create(mojo::PendingReceiver<ml::webnn::mojom::Graph> receiver,
                     scoped_refptr<ExecutionContext> execution_context,
                     uint32_t graph_id);

  GraphDMLImpl(const GraphDMLImpl&) = delete;
  GraphDMLImpl& operator=(const GraphDMLImpl&) = delete;

 protected:
  GraphDMLImpl(scoped_refptr<ExecutionContext> execution_context,
               uint32_t graph_id);

 private:
  // ml::webnn::mojom::Graph
  void AddInput(const std::string&, OperandDescriptorPtr) override;
  void AddConstant(OperandDescriptorPtr) override;
  void AddElementWiseBinary(uint32_t,
                            uint32_t,
                            BinaryOperandType,
                            OperandDescriptorPtr) override;
  void AddClamp(uint32_t input_id,
                ClampOptionsPtr options,
                OperandDescriptorPtr desc) override;
  void AddConv2d(uint32_t input_id,
                 uint32_t filter_id,
                 Conv2dOptionsPtr options,
                 OperandDescriptorPtr desc) override;
  void AddReshape(uint32_t input_id, OperandDescriptorPtr desc) override;
  void AddGemm(uint32_t,
               uint32_t,
               GemmOptionsPtr,
               OperandDescriptorPtr) override;
  void AddPool2d(uint32_t input_id,
                 Pool2dOptionsPtr options,
                 Pool2dType type,
                 OperandDescriptorPtr desc) override;
  void AddUnary(uint32_t input_id,
                UnaryOperandType type,
                OperandDescriptorPtr desc) override;
  void AddFusionClamp(ClampOptionsPtr options, uint32_t operator_id) override;

  void Build(const base::flat_map<std::string, uint32_t>& named_operands,
             ConstantsInfoPtr constants_info,
             BuildCallback callback) override;
  void Compute(NamedInputsPtr named_inputs, ComputeCallback callback) override;

  std::unique_ptr<NodeOutput> Clamp(NodeOutput* input_node,
                                    const ClampOptions* options);
  void EmulateFusedOperator(const FusionOperator* activation,
                            std::unique_ptr<NodeOutput>& input_node,
                            const std::vector<UINT>& inputDims);
  void TransposeOutputToNhwc(std::unique_ptr<NodeOutput>& input_node,
                             const std::vector<UINT>& nchwOutputDims);

  void AddOutput(const std::string&, uint32_t);

  uint32_t graph_id_;
  scoped_refptr<ExecutionContext> execution_context_;
  std::unique_ptr<UploadHeap> input_resource_uploader_;
  std::unique_ptr<ReadbackHeap> output_resource_readback_;
  std::unique_ptr<GraphDescBuilder> graph_desc_builder_;

  // IDMLCompiledOperator represents the DirectML graph's output which need to
  // be initialized by IDMLOperatorInitializer.
  ComPtr<IDMLCompiledOperator> mCompiledOperator;

  std::map<uint32_t, std::unique_ptr<NodeOutput>> node_output_map_;

  std::string error_messages_;
  BuildResult build_result_;
  std::unique_ptr<FusionOperators> fusion_operators_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_GRAPH_DML_IMPL_H_

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
using ml::webnn::mojom::Conv2dOptionsPtr;
using ml::webnn::mojom::FusionOperator;
using ml::webnn::mojom::GemmOptionsPtr;
using ml::webnn::mojom::NamedInputsPtr;
using ml::webnn::mojom::ConstantsInfoPtr;
using ml::webnn::mojom::NamedOutputsPtr;
using ml::webnn::mojom::OperandDescriptorPtr;
using ml::webnn::mojom::Pool2dOptions;
using ml::webnn::mojom::Pool2dOptionsPtr;
using ml::webnn::mojom::Pool2dType;
using ml::webnn::mojom::UnaryOperandType;

}  // namespace

class FusionOperators;

struct MemoryInfo {
  MemoryInfo();
  ~MemoryInfo();
  size_t byte_offset;
  size_t byte_length;
};

// Represent the DirectML tensor description.
struct DmlTensorDesc {
  DmlTensorDesc();
  ~DmlTensorDesc();
  std::vector<UINT> dimensions = {};
  std::vector<UINT> strides = {};
  // Describes a tensor that will be stored in a Direct3D 12 buffer resource.
  DML_BUFFER_TENSOR_DESC bufferDesc = {};
};

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
  GraphDMLImpl(scoped_refptr<ExecutionContext> execution_context, uint32_t graph_id);

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

  void AddEdgesToThisNode(
      std::vector<std::shared_ptr<EdgeInfoBase>> inputNodes);
  std::shared_ptr<EdgeInfoBase> Clamp(std::shared_ptr<EdgeInfoBase> inputEdge,
                                      const ClampOptions* options);
  void EmulateFusedOperator(const FusionOperator* activation,
                            std::shared_ptr<EdgeInfoBase>& inputEdge,
                            const std::vector<UINT>& inputDims);
  void TransposeOutputToNhwc(std::shared_ptr<EdgeInfoBase>& inputEdge,
                             const std::vector<UINT>& nchwOutputDims);

  BuildResult Finish();
  void AddOutput(const std::string&, uint32_t);

  scoped_refptr<ExecutionContext> execution_context_;
  uint32_t graph_id_;
  // Represents a DirectML device, which is used to create operators, binding
  // tables, command recorders, and other objects.
  ComPtr<IDMLDevice> mDevice;
  // The IDMLDevice1 interface inherits from IDMLDevice.
  ComPtr<IDMLDevice1> mDevice1;
  // Represents a virtual adapter; it is used to create command allocators,
  // command lists, command queues, fences, resources, pipeline state objects,
  // heaps, root signatures, samplers, and many resource views.
  ComPtr<ID3D12Device> mD3D12Device;

  ComPtr<IDMLCommandRecorder> mCommandRecorder;
  ComPtr<ID3D12CommandQueue> mCommandQueue;
  ComPtr<ID3D12CommandAllocator> mCommandAllocator;
  ComPtr<ID3D12GraphicsCommandList> mCommandList;

  UINT64 mOutputsResourceSize = 0;
  UINT64 mCommonInputsResourceSize = 0;
  ComPtr<ID3D12Resource> mUploadResource = nullptr;
  ComPtr<ID3D12Resource> mInputResource = nullptr;
  ComPtr<ID3D12Resource> mOutputResource = nullptr;
  ComPtr<ID3D12Resource> mReadBackResource = nullptr;
#ifdef ENABLE_GPU_MEMORY_MANAGEMENT
#else
  // std::vector<ComPtr<ID3D12Resource>> constants_resource_;
  std::vector<ComPtr<ID3D12Resource>> inputs_resource_;
#endif

  // Describe a graph of DirectML operators used to compile a combined,
  // optimized operator.
  std::vector<std::shared_ptr<InputEdgeInfo>> mInputs;
  std::vector<EdgeInfo> mOutputs;
  std::vector<DML_GRAPH_NODE_DESC> mIntermediateNodes;
  std::vector<DML_GRAPH_EDGE_DESC> mInputEdges;
  std::vector<DML_GRAPH_EDGE_DESC> mOutputEdges;
  std::vector<DML_GRAPH_EDGE_DESC> mIntermediateEdges;

  // IDMLCompiledOperator represents the DirectML graph's output which need to
  // be initialized by IDMLOperatorInitializer.
  ComPtr<IDMLCompiledOperator> mCompiledOperator;

  std::map<uint32_t, std::shared_ptr<EdgeInfoBase>> mGraphEdgesMap;

  // Keep intermediate nodes here to avoid releasing too early.
  std::map<uint32_t, ComPtr<IDMLOperator>> mIntermediateNodesMap;
  // Keep the input tensors description here to avoid releasing too early.
  std::vector<std::shared_ptr<DmlTensorDesc>> mDmlTensorsDesc;
  // Keep the descriptions of nodes and edges here to avoid releasing too early.
  std::vector<std::unique_ptr<DML_OPERATOR_GRAPH_NODE_DESC>>
      mIntermediateNodesDesc;
  std::vector<std::unique_ptr<DML_INPUT_GRAPH_EDGE_DESC>> mInputEdgesDesc;
  std::vector<std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC>> mOutputEdgesDesc;
  std::vector<std::unique_ptr<DML_INTERMEDIATE_GRAPH_EDGE_DESC>>
      mIntermediateEdgesDesc;

  std::string error_messages_;
  BuildResult build_result_;
  std::unique_ptr<FusionOperators> fusion_operators_;

  std::map<std::string, MemoryInfo> outputs_info_map_;
  // std::map<std::string, base::ReadOnlySharedMemoryMapping>
  // outputs_shm_mapping_;
  base::MappedReadOnlyRegion outputs_shm_region_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_GRAPH_DML_IMPL_H_

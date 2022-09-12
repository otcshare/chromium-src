// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/graph_dml_impl.h"

#include "base/logging.h"
#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/execution_context.h"
#include "content/browser/ml/webnn/dml/graph_dml_impl.h"
#include "content/browser/ml/webnn/dml/unordered_resources.h"
#include "content/browser/ml/webnn/dml/upload_heap.h"
#include "content/browser/ml/webnn/fusion_operators.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "utils_dml.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::AutoPad;
using ml::webnn::mojom::ConstantsInfoPtr;
using ml::webnn::mojom::Conv2dFilterOperandLayout;
using ml::webnn::mojom::Conv2dOptions;
using ml::webnn::mojom::Conv2dOptionsPtr;
using ml::webnn::mojom::FusionOperator;
using ml::webnn::mojom::FusionOperatorPtr;
using ml::webnn::mojom::FusionType;
using ml::webnn::mojom::InputOperandLayout;
using ml::webnn::mojom::MemoryInfoPtr;
using ml::webnn::mojom::OperandType;

enum TransposeType { NhwcToNchw, NchwToNhwc };

std::vector<UINT> transposeStrides(TransposeType transposeType,
                                   const std::vector<UINT>& inputDims) {
  UINT nStride = 0, cStride = 0, hStride = 0, wStride = 0;
  switch (transposeType) {
    case NhwcToNchw:
      nStride = inputDims[1] * inputDims[2] * inputDims[3];
      hStride = inputDims[2] * inputDims[3];
      wStride = inputDims[3];
      cStride = 1;
      return {nStride, cStride, hStride, wStride};
    case NchwToNhwc:
      nStride = inputDims[1] * inputDims[2] * inputDims[3];
      cStride = inputDims[2] * inputDims[3];
      hStride = inputDims[3];
      wStride = 1;
      return {nStride, hStride, wStride, cStride};
    default:
      assert(0);
      break;
  }
}

std::vector<UINT> transposeStridesToNchw(
    const std::vector<UINT>& inputDims,
    const DML_TENSOR_DESC& inputTensorDesc) {
  const DML_BUFFER_TENSOR_DESC* bufferDesc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(inputTensorDesc.Desc);
  assert(bufferDesc != nullptr && bufferDesc->DimensionCount == 4);
  auto* strides = bufferDesc->Strides;
  if (strides != nullptr) {
    return {strides[0], strides[3], strides[1], strides[2]};
  } else {
    return transposeStrides(NhwcToNchw, inputDims);
  }
}

DML_OPERATOR_DESC* CreateFusedOperator(
    const FusionOperator* activation,
    DML_ACTIVATION_LINEAR_OPERATOR_DESC& dmlActicationOperatorDesc,
    DML_OPERATOR_DESC& dmlFusedOperatorDesc) {
  if (activation == nullptr) {
    return nullptr;
  }

  dmlActicationOperatorDesc.InputTensor = nullptr;
  dmlActicationOperatorDesc.OutputTensor = nullptr;
  dmlActicationOperatorDesc.Alpha = 0.0;
  dmlActicationOperatorDesc.Beta = 0.0;
  switch (activation->fusion_type) {
    case FusionType::kRelu: {
      dmlFusedOperatorDesc.Type = DML_OPERATOR_ACTIVATION_RELU;
    } break;
    case FusionType::kClamp:
      return nullptr;
    default:
      LOG(ERROR) << "This fusion type is not supported.";
      assert(0);
  }
  dmlFusedOperatorDesc.Desc = &dmlActicationOperatorDesc;
  return &dmlFusedOperatorDesc;
}

// Strides are used to express broadcasting (by specifying a stride of 0) as
// well as padding. If Strides is not specified, each dimension in the tensor is
// considered to be contiguously packed, with no additional padding. The
// calculated strides refer to
// https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
std::vector<UINT> CalculateStridesForBroadcast(
    std::vector<UINT> originDims,
    std::vector<UINT> broadcastedDims,
    const DML_TENSOR_DESC& inputTensorDesc,
    size_t skipAxes = 0) {
  auto originRank = originDims.size(), broadcastedRank = broadcastedDims.size();
  if (originRank < skipAxes || originRank > broadcastedRank) {
    LOG(ERROR) << "Shapes are incompatible, broadcasting failed.";
    assert(0);
  }
  std::vector<bool> broadcastFlags(broadcastedRank, false);
  auto rankGap = broadcastedRank - originRank;
  for (size_t i = 0; i < rankGap; ++i) {
    broadcastFlags[i] = true;
  }
  for (size_t i = 0; i < originRank - skipAxes; ++i) {
    if (originDims[i] == 1 && broadcastedDims[rankGap + i] != 1) {
      broadcastFlags[rankGap + i] = true;
    }
  }

  for (size_t i = 0; i < broadcastedRank; ++i) {
    if (broadcastFlags[i]) {
      broadcastedDims[i] = 1;
    }
  }
  std::vector<UINT> strides(broadcastedRank);

  const DML_BUFFER_TENSOR_DESC* bufferDesc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(inputTensorDesc.Desc);
  assert(bufferDesc != nullptr &&
         broadcastedRank >= bufferDesc->DimensionCount);
  auto* existedStrides = bufferDesc->Strides;
  if (existedStrides != nullptr) {
    auto indexBegin = broadcastedRank - bufferDesc->DimensionCount;
    for (size_t i = 0, j = 0; i < broadcastedRank; ++i) {
      if (i < indexBegin) {
        strides[i] = 0;
      } else {
        strides[i] = broadcastFlags[i] ? 0 : existedStrides[j];
        ++j;
      }
    }
  } else {
    strides[broadcastedRank - 1] = broadcastFlags[broadcastedRank - 1] ? 0 : 1;
    size_t elements = 1;
    for (size_t i = 1; i < broadcastedRank; i++) {
      size_t j = broadcastedRank - i - 1;
      elements *= broadcastedDims[j + 1];
      strides[j] = broadcastFlags[j] ? 0 : elements;
    }
  }
  return strides;
}

// uint32_t SizeOfShape(const std::vector<UINT>& dims) {
//   uint32_t prod = 1;
//   for (size_t i = 0; i < dims.size(); ++i)
//     prod *= dims[i];
//   return prod;
// }

std::vector<UINT> Dimensions(std::shared_ptr<EdgeInfoBase> edge) {
  const DML_BUFFER_TENSOR_DESC* desc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
          edge->outputTensorDESC.Desc);

  // std::vector<UINT> dimensions;
  // dimensions.assign(desc->Sizes, desc->Sizes + desc->DimensionCount);
  return std::vector<UINT>(desc->Sizes, desc->Sizes + desc->DimensionCount);
}

std::vector<UINT> ConvertDimensions(const std::vector<int32_t>& dimensions) {
  std::vector<UINT> convertedDimensions;
  for (auto dim : dimensions) {
    if (dim < 0) {
      LOG(ERROR) << "DML doesn't support the negative dimension value";
      assert(0);
    }
    convertedDimensions.push_back(dim);
  }
  return convertedDimensions;
}

std::vector<UINT> ExpandDimensions(const std::vector<UINT>& dims, size_t rank) {
  assert(rank >= dims.size());
  std::vector<UINT> newDims(rank, 1);
  for (size_t i = 0; i < dims.size(); ++i) {
    newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
  }
  return newDims;
}

std::vector<UINT> transposeDimensions(TransposeType transposeType,
                                      const std::vector<UINT>& inputDims) {
  std::vector<UINT> newInputDims(4);
  switch (transposeType) {
    case NhwcToNchw:
      newInputDims[0] = inputDims[0];
      newInputDims[1] = inputDims[3];
      newInputDims[2] = inputDims[1];
      newInputDims[3] = inputDims[2];
      break;
    case NchwToNhwc:
      newInputDims[0] = inputDims[0];
      newInputDims[1] = inputDims[2];
      newInputDims[2] = inputDims[3];
      newInputDims[3] = inputDims[1];
      break;
    default:
      assert(0);
      break;
  }
  return newInputDims;
}

std::vector<UINT> transposeFilterDimensionsAsOihw(
    Conv2dFilterOperandLayout filterLayout,
    const std::vector<UINT>& filterDims) {
  std::vector<UINT> newFilterDims(4);
  switch (filterLayout) {
    case Conv2dFilterOperandLayout::kOhwi:
      newFilterDims.resize(4);
      newFilterDims[0] = filterDims[0];
      newFilterDims[1] = filterDims[3];
      newFilterDims[2] = filterDims[1];
      newFilterDims[3] = filterDims[2];
      break;
    case Conv2dFilterOperandLayout::kHwio:
      newFilterDims[0] = filterDims[3];
      newFilterDims[1] = filterDims[2];
      newFilterDims[2] = filterDims[0];
      newFilterDims[3] = filterDims[1];
      break;
    case Conv2dFilterOperandLayout::kIhwo:
      newFilterDims[0] = filterDims[3];
      newFilterDims[1] = filterDims[0];
      newFilterDims[2] = filterDims[1];
      newFilterDims[3] = filterDims[2];
      break;
    default:
      assert(0);
      break;
  }
  return newFilterDims;
}

std::vector<UINT> transposeFilterStridesAsOihw(
    Conv2dFilterOperandLayout filterLayout,
    const std::vector<UINT>& filterDims) {
  UINT hStride = 0, wStride = 0, iStride = 0, oStride = 0;
  switch (filterLayout) {
    case Conv2dFilterOperandLayout::kHwio:
      hStride = filterDims[1] * filterDims[2] * filterDims[3];
      wStride = filterDims[2] * filterDims[3];
      iStride = filterDims[3];
      oStride = 1;
      break;
    case Conv2dFilterOperandLayout::kOhwi:
      oStride = filterDims[1] * filterDims[2] * filterDims[3];
      hStride = filterDims[2] * filterDims[3];
      wStride = filterDims[3];
      iStride = 1;
      break;
    case Conv2dFilterOperandLayout::kIhwo:
      iStride = filterDims[1] * filterDims[2] * filterDims[3];
      hStride = filterDims[2] * filterDims[3];
      wStride = filterDims[3];
      oStride = 1;
      break;
    default:
      assert(0);
      break;
  }
  return {oStride, iStride, hStride, wStride};
}

template <typename T>
std::vector<UINT> ImplicitPadding(const T* options,
                                  const std::vector<UINT>& inputDims,
                                  const std::vector<UINT>& filterDims) {
  return ComputeImplicitPaddingForAutoPad<T, UINT>(
      options, {inputDims[2], inputDims[3]},
      {filterDims[filterDims.size() - 2], filterDims[filterDims.size() - 1]});
}

template <typename T>
std::vector<UINT> ExplicitPadding(const T* options) {
  UINT paddingTop = static_cast<UINT>(options->padding[0]);
  UINT paddingBottom = static_cast<UINT>(options->padding[1]);
  UINT paddingLeft = static_cast<UINT>(options->padding[2]);
  UINT paddingRight = static_cast<UINT>(options->padding[3]);

  return {paddingTop, paddingBottom, paddingLeft, paddingRight};
}

bool CreateDmlTensorDesc(
    std::vector<std::shared_ptr<DmlTensorDesc>>& dmlTensorsDesc,
    const std::shared_ptr<DmlTensorDesc>& dmlTensorDesc,
    const std::vector<UINT>& dimensions,
    const std::vector<UINT>& strides = {},
    DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32,
    DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAG_NONE) {
  dmlTensorDesc->dimensions = dimensions;
  dmlTensorDesc->strides = strides;
  if (!strides.empty() && dimensions.size() != strides.size()) {
    LOG(ERROR) << "Dimension size should be equal to strides size.";
    return false;
  }

  size_t typeLength = 4;
  switch (dataType) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_INT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
      break;
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      typeLength = 2;
      break;
    default:
      LOG(ERROR) << "This data type is not supported";
      return false;
  }

  size_t elementsCount = 1;
  if (dmlTensorDesc->dimensions.size() > DML_TENSOR_DIMENSION_COUNT_MAX) {
    LOG(ERROR) << "Tensor dimension count " << dmlTensorDesc->dimensions.size()
               << " is greater than DML_TENSOR_DIMENSION_COUNT_MAX "
               << DML_TENSOR_DIMENSION_COUNT_MAX;
    return false;
  }
  if (dmlTensorDesc->dimensions.size() == 0) {
    dmlTensorDesc->dimensions.resize(1);
    dmlTensorDesc->dimensions[0] = 1;
  } else {
    for (uint32_t i = 0; i < dmlTensorDesc->dimensions.size(); ++i) {
      auto dim = dmlTensorDesc->dimensions[i];
      if (strides.empty()) {
        elementsCount *= dim;
      } else {
        // The specific dim from broadcasting shouldn't increase the count of
        // elements.
        if (strides[i] == 0) {
          dim = 1;
        }
        elementsCount *= dim;
      }
    }
  }
  auto TotalTensorSizeInBytes = elementsCount * typeLength;
  dmlTensorDesc->bufferDesc.DimensionCount = dmlTensorDesc->dimensions.size();
  dmlTensorDesc->bufferDesc.Sizes = dmlTensorDesc->dimensions.data();
  dmlTensorDesc->bufferDesc.Strides = dmlTensorDesc->strides.data();
  dmlTensorDesc->bufferDesc.TotalTensorSizeInBytes = TotalTensorSizeInBytes;
  dmlTensorDesc->bufferDesc.GuaranteedBaseOffsetAlignment = 0;
  dmlTensorDesc->bufferDesc.DataType = dataType;
  dmlTensorDesc->bufferDesc.Flags = tensorFlag;

  dmlTensorsDesc.push_back(dmlTensorDesc);
  return true;
}

bool CreateDmlTensorDesc(
    std::vector<std::shared_ptr<DmlTensorDesc>>& dmlTensorsDesc,
    const std::shared_ptr<DmlTensorDesc>& dmlTensorDesc,
    OperandDescriptorPtr& desc,
    DML_TENSOR_FLAGS tensorFlag = DML_TENSOR_FLAGS::DML_TENSOR_FLAG_NONE) {
  std::vector<UINT> dimensions = ConvertDimensions(desc->dimensions);
  DML_TENSOR_DATA_TYPE dataType;
  // for (uint32_t i = 0; i < desc->dimensionsCount; ++i) {
  //     if (desc->dimensions[i] < 0) {
  //         LOG(ERROR) << "DML doesn't support the negative dimension value";
  //         return false;
  //     }
  // }
  // dimensions.assign(desc->dimensions, desc->dimensions +
  // desc->dimensionsCount);
  if (desc->data_type == OperandType::kFloat32) {
    dataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if (desc->data_type == OperandType::kFloat16) {
    dataType = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if (desc->data_type == OperandType::kInt32) {
    dataType = DML_TENSOR_DATA_TYPE_INT32;
  } else if (desc->data_type == OperandType::kUint32) {
    dataType = DML_TENSOR_DATA_TYPE_UINT32;
  } else {
    LOG(ERROR) << "This data type is not supported";
    return false;
  }

  return CreateDmlTensorDesc(dmlTensorsDesc, dmlTensorDesc, dimensions, {},
                             dataType, tensorFlag);
}

bool CreateDmlTensorDesc(
    std::vector<std::shared_ptr<DmlTensorDesc>>& dmlTensorsDesc,
    const std::shared_ptr<DmlTensorDesc>& dmlTensorDesc,
    DML_TENSOR_DESC* tensorDESC,
    std::vector<UINT> dimensions = {},
    std::vector<UINT> strides = {},
    bool useDefaultFlags = false) {
  assert(tensorDESC != nullptr);
  const DML_BUFFER_TENSOR_DESC* desc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(tensorDESC->Desc);

  if (dimensions.empty()) {
    dimensions.assign(desc->Sizes, desc->Sizes + desc->DimensionCount);
  }
  DML_TENSOR_FLAGS tensorFlags =
      useDefaultFlags ? DML_TENSOR_FLAG_NONE : desc->Flags;
  return CreateDmlTensorDesc(dmlTensorsDesc, dmlTensorDesc, dimensions, strides,
                             desc->DataType, tensorFlags);
}

// Only used to create the output edge from a node.
std::shared_ptr<EdgeInfoBase> CreateEdgeFromThisNode(
    const DML_TENSOR_DESC& outputTensorDesc,
    const uint32_t nodeIndex,
    const uint32_t outputNodeIndex = 0,
    bool isDefault = true) {
  std::shared_ptr<EdgeInfo> edgeInfo(new EdgeInfo());
  edgeInfo->outputTensorDESC = outputTensorDesc;
  edgeInfo->nodeIndex = nodeIndex;
  edgeInfo->outputNodeIndex = outputNodeIndex;
  edgeInfo->isInputEdge = false;
  std::shared_ptr<EdgeInfoBase> edge(edgeInfo);
  return edge;
}

std::shared_ptr<EdgeInfoBase> updateEdge(std::shared_ptr<EdgeInfoBase> edge,
                                         const DML_TENSOR_DESC& tensorDesc) {
  if (edge->isInputEdge) {
    std::shared_ptr<InputEdgeInfo> newEdgeInfo(new InputEdgeInfo());
    memcpy(static_cast<void*>(newEdgeInfo.get()),
           static_cast<void*>(edge.get()), sizeof(InputEdgeInfo));
    newEdgeInfo->outputTensorDESC = tensorDesc;
    std::shared_ptr<EdgeInfoBase> newEdge(newEdgeInfo);
    return newEdge;
  } else {
    std::shared_ptr<EdgeInfo> newEdgeInfo(new EdgeInfo());
    memcpy(static_cast<void*>(newEdgeInfo.get()),
           static_cast<void*>(edge.get()), sizeof(EdgeInfo));
    newEdgeInfo->outputTensorDESC = tensorDesc;
    std::shared_ptr<EdgeInfoBase> newEdge(newEdgeInfo);
    return newEdge;
  }
}

}  // namespace

DmlTensorDesc::DmlTensorDesc() = default;
DmlTensorDesc::~DmlTensorDesc() = default;

MemoryInfo::MemoryInfo() = default;
MemoryInfo::~MemoryInfo() = default;

InputEdgeInfo::InputEdgeInfo() = default;
InputEdgeInfo::~InputEdgeInfo() = default;

#define DAWN_INTERNAL_ERROR(MESSAGE)            \
  do {                                          \
    error_messages_ = MESSAGE;                  \
    assert(0);                                  \
    build_result_ = BuildResult::kUnknownError; \
    return;                                     \
  } while (0)

#define CREATE_OPERATOR(type, dmlSpecificOperatorDesc) \
  DML_OPERATOR_DESC dmlOperatorDesc = {};              \
  dmlOperatorDesc.Type = DML_OPERATOR_##type;          \
  dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;     \
  WEBNN_CHECK(                                         \
      mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));

#define CREATE_BINARY_OPERATOR(type, aTensorDesc, bTensorDesc,       \
                               outputTensorDesc, dmlOperator)        \
  DML_ELEMENT_WISE_##type##_OPERATOR_DESC dmlSpecificOperatorDesc{}; \
  dmlSpecificOperatorDesc.ATensor = &aTensorDesc;                    \
  dmlSpecificOperatorDesc.BTensor = &bTensorDesc;                    \
  dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;          \
  CREATE_OPERATOR(ELEMENT_WISE_##type, dmlSpecificOperatorDesc)

#define CREATE_UNARY_OPERATOR(type, inputTensorDesc, dmlOperator) \
  DML_##type##_OPERATOR_DESC dmlSpecificOperatorDesc{};           \
  dmlSpecificOperatorDesc.InputTensor = &inputTensorDesc;         \
  dmlSpecificOperatorDesc.OutputTensor = &inputTensorDesc;        \
  CREATE_OPERATOR(type, dmlSpecificOperatorDesc)

// Append IDENTITY to remove the strides of input tensor. Use this to implement
// Reshape, Squeeze, Transpose and avoid creating an invaild graph with input =
// output.
#define APPEND_IDENTITY(inputTensorDesc, outputTensorDesc, dmlOperator)       \
  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC dmlSpecificOperatorDesc{};          \
  dmlSpecificOperatorDesc.InputTensor = &inputTensorDesc;                     \
  dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;                   \
  DML_OPERATOR_DESC dmlOperatorDesc = {};                                     \
  dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;                  \
  dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;                            \
  WEBNN_CHECK(                                                                \
      mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator))); \
  mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

// static
void GraphDMLImpl::Create(mojo::PendingReceiver<Graph> receiver,
                          scoped_refptr<ExecutionContext> execution_context,
                          uint32_t graph_id) {
  mojo::MakeSelfOwnedReceiver<Graph>(
      base::WrapUnique(new GraphDMLImpl(execution_context, graph_id)),
      std::move(receiver));
}

GraphDMLImpl::~GraphDMLImpl() = default;

GraphDMLImpl::GraphDMLImpl(scoped_refptr<ExecutionContext> execution_context,
                           uint32_t graph_id)
    : execution_context_(execution_context),
      graph_id_(graph_id),
      fusion_operators_(std::make_unique<FusionOperators>()) {
  mCommandQueue = execution_context->GetCommandQueue();
  mCommandAllocator = execution_context->GetCommandAllocator();
  mCommandList = execution_context->GetCommandList();
  WEBNN_CHECK(mCommandQueue->GetDevice(IID_PPV_ARGS(&mD3D12Device)));
  mDevice = execution_context->GetDMLDevice();
}

void GraphDMLImpl::AddInput(const std::string& name,
                            OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  // const OperandDescriptor* desc = input->GetOperandDescriptor();
  std::shared_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, dmlTensorDesc, desc)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC outputTensorDESC = {DML_TENSOR_TYPE_BUFFER,
                                      &(dmlTensorDesc->bufferDesc)};
  std::shared_ptr<InputEdgeInfo> inputEdgeInfo(new InputEdgeInfo());
  inputEdgeInfo->outputTensorDESC = outputTensorDESC;
  inputEdgeInfo->name = name;
  inputEdgeInfo->isInputEdge = true;
  inputEdgeInfo->inputIndex = mInputs.size();
  inputEdgeInfo->byteLength = dmlTensorDesc->bufferDesc.TotalTensorSizeInBytes;
  inputEdgeInfo->object_id = desc->object_id;
  std::shared_ptr<EdgeInfoBase> edge(inputEdgeInfo);

  mGraphEdgesMap[desc->object_id] = edge;
  mInputs.push_back(inputEdgeInfo);
  return;
}

void GraphDMLImpl::AddConstant(OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  if (mGraphEdgesMap.find(desc->object_id) != mGraphEdgesMap.end()) {
    LOG(ERROR) << "=======There are issues in sorting graph";
    return;
  }
  std::shared_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, dmlTensorDesc, desc,
                           DML_TENSOR_FLAG_OWNED_BY_DML)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC outputTensorDESC = {DML_TENSOR_TYPE_BUFFER,
                                      &(dmlTensorDesc->bufferDesc)};

  std::shared_ptr<InputEdgeInfo> inputEdgeInfo(new InputEdgeInfo());
  inputEdgeInfo->outputTensorDESC = outputTensorDESC;
  inputEdgeInfo->name = "Input_Constant_" + std::to_string(mInputs.size());
  inputEdgeInfo->isInputEdge = true;
  inputEdgeInfo->inputIndex = mInputs.size();
  // Keep the resource until initializing operators
  // constants_resource_.push_back(resource);
  // inputEdgeInfo->resource = std::move(resource);
  inputEdgeInfo->isConstantInput = true;
  inputEdgeInfo->object_id = desc->object_id;
  std::shared_ptr<EdgeInfoBase> edge(inputEdgeInfo);

  mGraphEdgesMap[desc->object_id] = edge;
  mInputs.push_back(inputEdgeInfo);
  return;
}

// Add an intermediate node to the graph, and add the related input edges(if
// exist) and intermediate edges to this node by the way.
void GraphDMLImpl::AddEdgesToThisNode(
    std::vector<std::shared_ptr<EdgeInfoBase>> edges) {
  std::unique_ptr<DML_OPERATOR_GRAPH_NODE_DESC> nodeDesc(
      new DML_OPERATOR_GRAPH_NODE_DESC);
  nodeDesc->Operator = mIntermediateNodesMap[mIntermediateNodes.size()].Get();

  for (size_t i = 0; i < edges.size(); ++i) {
    if (edges[i]->isInputEdge) {
      auto* edge = reinterpret_cast<InputEdgeInfo*>(edges[i].get());
      std::unique_ptr<DML_INPUT_GRAPH_EDGE_DESC> inputEdgeDesc(
          new DML_INPUT_GRAPH_EDGE_DESC);
      inputEdgeDesc->GraphInputIndex = edge->inputIndex;
      inputEdgeDesc->ToNodeIndex = mIntermediateNodes.size();
      inputEdgeDesc->ToNodeInputIndex = i;
      mInputEdges.push_back({DML_GRAPH_EDGE_TYPE_INPUT, inputEdgeDesc.get()});
      mInputEdgesDesc.push_back(std::move(inputEdgeDesc));
    } else {
      auto* edge = reinterpret_cast<EdgeInfo*>(edges[i].get());
      std::unique_ptr<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdgeDesc(
          new DML_INTERMEDIATE_GRAPH_EDGE_DESC);
      intermediateEdgeDesc->FromNodeIndex = edge->nodeIndex;
      intermediateEdgeDesc->FromNodeOutputIndex = edge->outputNodeIndex;
      intermediateEdgeDesc->ToNodeIndex = mIntermediateNodes.size();
      intermediateEdgeDesc->ToNodeInputIndex = i;
      mIntermediateEdges.push_back(
          {DML_GRAPH_EDGE_TYPE_INTERMEDIATE, intermediateEdgeDesc.get()});
      mIntermediateEdgesDesc.push_back(std::move(intermediateEdgeDesc));
    }
  }
  mIntermediateNodes.push_back({DML_GRAPH_NODE_TYPE_OPERATOR, nodeDesc.get()});
  mIntermediateNodesDesc.push_back(std::move(nodeDesc));
}

void GraphDMLImpl::AddElementWiseBinary(uint32_t a_id,
                                        uint32_t b_id,
                                        BinaryOperandType type,
                                        OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(mGraphEdgesMap.find(a_id) != mGraphEdgesMap.end());
  assert(mGraphEdgesMap.find(b_id) != mGraphEdgesMap.end());

  auto aEdge = mGraphEdgesMap[a_id];
  auto bEdge = mGraphEdgesMap[b_id];
  auto aDims = Dimensions(aEdge);
  auto bDims = Dimensions(bEdge);
  auto outputDims = ConvertDimensions(output_desc->dimensions);
  size_t aRank = aDims.size(), bRank = bDims.size(),
         outputRank = outputDims.size();
  size_t broadcastSkipAxis = 0;
  std::vector<UINT> aNewDims, bNewDims, outputNewDims = outputDims;

  if (type == BinaryOperandType::kMatMul) {
    // DML GEMM requires 4D input tensors.
    if (aRank > 4 || bRank > 4) {
      DAWN_INTERNAL_ERROR("The size of input dimensions is greater than 4.");
    }
    if (aRank < 4) {
      aDims = ExpandDimensions(aDims, 4);
    }

    if (bRank < 4) {
      if (bRank == 1) {
        // If b is 1-D, it is converted to a 2-D tensor by by appending a 1 to
        // its dimensions.
        bDims.push_back(1);
      }
      bDims = ExpandDimensions(bDims, 4);
    }

    if (outputRank < 4) {
      outputNewDims = ExpandDimensions(outputDims, 4);
    }

    if (aRank > 2 || bRank > 2) {
      // If either a or b is N-D, N > 2, it is treated as a stack of matrices
      // with dimensions corresponding to the last two indices. The matrix
      // multiplication will be broadcasted accordingly by following
      // [numpy-broadcasting-rule].
      broadcastSkipAxis = 2;
    }
    aNewDims = bNewDims = outputNewDims;
    aNewDims[2] = aDims[2];
    aNewDims[3] = aDims[3];
    bNewDims[2] = bDims[2];
    bNewDims[3] = bDims[3];
  } else {
    aNewDims = bNewDims = outputNewDims;
  }

  auto aNewStrides = CalculateStridesForBroadcast(
      aDims, aNewDims, aEdge->outputTensorDESC, broadcastSkipAxis);
  std::shared_ptr<DmlTensorDesc> aDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, aDmlTensorDesc,
                           &aEdge->outputTensorDESC, aNewDims, aNewStrides)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC aTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                 &aDmlTensorDesc->bufferDesc};

  auto bNewStrides = CalculateStridesForBroadcast(
      bDims, bNewDims, bEdge->outputTensorDESC, broadcastSkipAxis);
  std::shared_ptr<DmlTensorDesc> bDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, bDmlTensorDesc,
                           &bEdge->outputTensorDESC, bNewDims, bNewStrides)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC bTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                 &bDmlTensorDesc->bufferDesc};

  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &aEdge->outputTensorDESC, outputNewDims, {}, true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                      &outputDmlTensorDesc->bufferDesc};

  ComPtr<IDMLOperator> dmlOperator;
  switch (type) {
    case BinaryOperandType::kAdd: {
      CREATE_BINARY_OPERATOR(ADD, aTensorDesc, bTensorDesc, outputTensorDesc,
                             dmlOperator);
    } break;
    case BinaryOperandType::kDiv: {
      CREATE_BINARY_OPERATOR(DIVIDE, aTensorDesc, bTensorDesc, outputTensorDesc,
                             dmlOperator);
    } break;
    case BinaryOperandType::kMul: {
      CREATE_BINARY_OPERATOR(MULTIPLY, aTensorDesc, bTensorDesc,
                             outputTensorDesc, dmlOperator);
    } break;
    case BinaryOperandType::kSub: {
      CREATE_BINARY_OPERATOR(SUBTRACT, aTensorDesc, bTensorDesc,
                             outputTensorDesc, dmlOperator);
    } break;
    case BinaryOperandType::kMax: {
      CREATE_BINARY_OPERATOR(MAX, aTensorDesc, bTensorDesc, outputTensorDesc,
                             dmlOperator);
    } break;
    case BinaryOperandType::kMin: {
      CREATE_BINARY_OPERATOR(MIN, aTensorDesc, bTensorDesc, outputTensorDesc,
                             dmlOperator);
    } break;
    case BinaryOperandType::kPower: {
      DML_ELEMENT_WISE_POW_OPERATOR_DESC dmlSpecificOperatorDesc{};
      dmlSpecificOperatorDesc.InputTensor = &aTensorDesc;
      dmlSpecificOperatorDesc.ExponentTensor = &bTensorDesc;
      dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;
      DML_OPERATOR_DESC dmlOperatorDesc = {};
      dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_POW;
      dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;
      WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc,
                                          IID_PPV_ARGS(&dmlOperator)));
    } break;
    case BinaryOperandType::kMatMul: {
      DML_GEMM_OPERATOR_DESC dmlSpecificOperatorDesc{};
      dmlSpecificOperatorDesc.ATensor = &aTensorDesc;
      dmlSpecificOperatorDesc.BTensor = &bTensorDesc;
      dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;
      dmlSpecificOperatorDesc.Alpha = 1.0;
      DML_OPERATOR_DESC dmlOperatorDesc = {};
      dmlOperatorDesc.Type = DML_OPERATOR_GEMM;
      dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;
      WEBNN_CHECK(mDevice->CreateOperator(&dmlOperatorDesc,
                                          IID_PPV_ARGS(&dmlOperator)));
    } break;
    default:
      DAWN_INTERNAL_ERROR(" Binary op is not implemented.");
  }
  if (outputDims != outputNewDims) {
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                             &aEdge->outputTensorDESC, outputDims, {}, true)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
  }

  mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
  mGraphEdgesMap[output_desc->object_id] =
      CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
  AddEdgesToThisNode({aEdge, bEdge});
  return;
}

std::shared_ptr<EdgeInfoBase> GraphDMLImpl::Clamp(
    std::shared_ptr<EdgeInfoBase> inputEdge,
    const ClampOptions* options) {
  DML_TENSOR_DESC inputTensorDesc = inputEdge->outputTensorDESC;

  // Set OutputTensor = InputTensor with the same strides to optimize
  // performance.
  DML_ELEMENT_WISE_CLIP_OPERATOR_DESC desc = {};
  desc.InputTensor = &inputTensorDesc;
  desc.OutputTensor = &inputTensorDesc;
  desc.ScaleBias = nullptr;
  desc.Min = options->minValue;
  desc.Max = options->maxValue;
  DML_OPERATOR_DESC dmlOperatorDesc = {};
  dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_CLIP;
  dmlOperatorDesc.Desc = &desc;

  ComPtr<IDMLOperator> dmlOperator;
  WEBNN_CHECK(
      mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
  mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

  std::shared_ptr<EdgeInfoBase> outputEdge =
      CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
  AddEdgesToThisNode({inputEdge});
  return outputEdge;
}

void GraphDMLImpl::AddClamp(uint32_t input_id,
                            ClampOptionsPtr options,
                            OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  auto inputEdge = mGraphEdgesMap[input_id];
  mGraphEdgesMap[output_desc->object_id] = Clamp(inputEdge, options.get());
  return;
}

void GraphDMLImpl::EmulateFusedOperator(
    const FusionOperator* activation,
    std::shared_ptr<EdgeInfoBase>& inputEdge,
    const std::vector<UINT>& inputDims) {
  // HardSwish and Clamp are not supported for fusion, so we add
  // them directly to
  // emulate. Currently we implement Relu6 operator by Clamp.
  if (activation == nullptr) {
    return;
  }

  auto fusionType = activation->fusion_type;
  if (fusionType == FusionType::kClamp) {
    auto* options = fusion_operators_->GetClampOption(activation->object_id);
    inputEdge = Clamp(inputEdge, options);
  }
  return;
}

void GraphDMLImpl::TransposeOutputToNhwc(
    std::shared_ptr<EdgeInfoBase>& inputEdge,
    const std::vector<UINT>& nchwOutputDims) {
  auto nhwcOutputStrides = transposeStrides(NchwToNhwc, nchwOutputDims);
  auto nhwcOutputDims = transposeDimensions(NchwToNhwc, nchwOutputDims);
  std::shared_ptr<DmlTensorDesc> nhwcOutputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, nhwcOutputDmlTensorDesc,
                           &inputEdge->outputTensorDESC, nhwcOutputDims,
                           nhwcOutputStrides, true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC nhwcOutputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                          &nhwcOutputDmlTensorDesc->bufferDesc};

  inputEdge = updateEdge(inputEdge, nhwcOutputTensorDesc);
  return;
}

void GraphDMLImpl::AddConv2d(uint32_t input_id,
                             uint32_t filter_id,
                             Conv2dOptionsPtr options,
                             OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(mGraphEdgesMap.find(input_id) != mGraphEdgesMap.end());
  assert(mGraphEdgesMap.find(filter_id) != mGraphEdgesMap.end());

  auto inputEdge = mGraphEdgesMap[input_id];
  auto filterEdge = mGraphEdgesMap[filter_id];

  auto inputDims = Dimensions(inputEdge);
  auto filterDims = Dimensions(filterEdge);
  auto outputDims = ConvertDimensions(output_desc->dimensions);
  std::vector<UINT> newInputDims = inputDims, newFilterDims = filterDims,
                    newOutputDims = outputDims, newInputStrides,
                    newFilterStrides;

  DML_TENSOR_DESC inputTensorDesc = inputEdge->outputTensorDESC;
  if (options->inputLayout == InputOperandLayout::kNhwc) {
    newInputDims = transposeDimensions(NhwcToNchw, inputDims);
    newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
    newInputStrides = transposeStridesToNchw(inputDims, inputTensorDesc);

    std::shared_ptr<DmlTensorDesc> inputDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, inputDmlTensorDesc,
                             &inputEdge->outputTensorDESC, newInputDims,
                             newInputStrides)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    inputTensorDesc = {DML_TENSOR_TYPE_BUFFER, &inputDmlTensorDesc->bufferDesc};
  }

  DML_TENSOR_DESC filterTensorDesc = filterEdge->outputTensorDESC;
  if (options->filterLayout != Conv2dFilterOperandLayout::kOihw) {
    newFilterDims =
        transposeFilterDimensionsAsOihw(options->filterLayout, filterDims);
    newFilterStrides =
        transposeFilterStridesAsOihw(options->filterLayout, filterDims);

    std::shared_ptr<DmlTensorDesc> filterDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, filterDmlTensorDesc,
                             &filterEdge->outputTensorDESC, newFilterDims,
                             newFilterStrides)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    filterTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                        &filterDmlTensorDesc->bufferDesc};
  }

  std::vector<std::shared_ptr<EdgeInfoBase>> inputEdges = {inputEdge,
                                                           filterEdge};

  const DML_TENSOR_DESC* biasTensorDescPtr = nullptr;
  DML_TENSOR_DESC newBiasTensorDesc = {};
  if (options->bias_id != 0) {
    assert(mGraphEdgesMap.find(options->bias_id) != mGraphEdgesMap.end());
    auto biasEdge = mGraphEdgesMap[options->bias_id];
    auto biasDims = Dimensions(biasEdge);
    if (biasDims[0] != newFilterDims[0] || biasDims.size() != 1) {
      DAWN_INTERNAL_ERROR(
          "The bias should be 1-D tensor with the shape of [output_channels].");
    }

    // Reshape bias from 1-D to 4-D for NCHW layout.
    std::vector<UINT> newBiasDims = {1, biasDims[0], 1, 1};
    std::shared_ptr<DmlTensorDesc> biasDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, biasDmlTensorDesc,
                             &biasEdge->outputTensorDESC, newBiasDims)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    newBiasTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                         &biasDmlTensorDesc->bufferDesc};
    biasTensorDescPtr = &newBiasTensorDesc;
    inputEdges.push_back(biasEdge);
  }

  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &inputEdge->outputTensorDESC, newOutputDims, {},
                           true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                      &outputDmlTensorDesc->bufferDesc};

  // FIXME(nhu): strides, dilations, padding should be uint32_t
  // need to fix the spec.
  std::vector<UINT> strides = ConvertDimensions(options->strides);
  std::vector<UINT> dilations = ConvertDimensions(options->dilations);

  std::vector<UINT> padding =
      options->auto_pad == AutoPad::kExplicit
          ? ExplicitPadding<Conv2dOptions>(options.get())
          : ImplicitPadding<Conv2dOptions>(options.get(), newInputDims,
                                           newFilterDims);
  std::vector<UINT> startPadding = {padding[0], padding[2]};
  std::vector<UINT> endPadding = {padding[1], padding[3]};
  std::vector<UINT> defaultOutPadding = {0, 0};

  DML_ACTIVATION_LINEAR_OPERATOR_DESC dmlActicationOperatorDesc{};
  DML_OPERATOR_DESC dmlFusedOperatorDesc = {};
  DML_OPERATOR_DESC* fusedActivation =
      CreateFusedOperator(options->activation.get(), dmlActicationOperatorDesc,
                          dmlFusedOperatorDesc);

  ComPtr<IDMLOperator> dmlOperator;
  DML_CONVOLUTION_OPERATOR_DESC dmlSpecificOperatorDesc{};
  dmlSpecificOperatorDesc.InputTensor = &inputTensorDesc;
  dmlSpecificOperatorDesc.FilterTensor = &filterTensorDesc;
  dmlSpecificOperatorDesc.BiasTensor = biasTensorDescPtr;
  dmlSpecificOperatorDesc.OutputTensor = &outputTensorDesc;

  dmlSpecificOperatorDesc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
  dmlSpecificOperatorDesc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
  dmlSpecificOperatorDesc.DimensionCount = inputDims.size() - 2;
  dmlSpecificOperatorDesc.Strides = strides.data();
  dmlSpecificOperatorDesc.Dilations = dilations.data();
  dmlSpecificOperatorDesc.StartPadding = startPadding.data();
  dmlSpecificOperatorDesc.EndPadding = endPadding.data();
  dmlSpecificOperatorDesc.OutputPadding = defaultOutPadding.data();
  dmlSpecificOperatorDesc.GroupCount = static_cast<UINT>(options->groups);
  dmlSpecificOperatorDesc.FusedActivation = fusedActivation;

  DML_OPERATOR_DESC dmlOperatorDesc = {};
  dmlOperatorDesc.Type = DML_OPERATOR_CONVOLUTION;
  dmlOperatorDesc.Desc = &dmlSpecificOperatorDesc;
  WEBNN_CHECK(
      mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
  mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

  auto outputEdge =
      CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
  AddEdgesToThisNode(inputEdges);

  // Transpose output from nchw->nhwc.
  if (options->inputLayout == InputOperandLayout::kNhwc) {
    TransposeOutputToNhwc(outputEdge, newOutputDims);
  }

  EmulateFusedOperator(options->activation.get(), outputEdge, outputDims);
  mGraphEdgesMap[output_desc->object_id] = outputEdge;
  return;
}

void GraphDMLImpl::AddReshape(uint32_t input_id,
                              OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(mGraphEdgesMap.find(input_id) != mGraphEdgesMap.end());

  auto inputEdge = mGraphEdgesMap[input_id];
  auto outputDims = ConvertDimensions(output_desc->dimensions);
  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  // Reshape needn't new strides, because the layout has not been changed.
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &inputEdge->outputTensorDESC, outputDims)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                      &outputDmlTensorDesc->bufferDesc};
  // Reshape is not a real node in DML, just need to update the edge created
  // from it.
  mGraphEdgesMap[output_desc->object_id] =
      updateEdge(inputEdge, outputTensorDesc);
  return;
}

void GraphDMLImpl::AddGemm(uint32_t a_id,
                           uint32_t b_id,
                           GemmOptionsPtr options,
                           OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(mGraphEdgesMap.find(a_id) != mGraphEdgesMap.end());
  auto aEdge = mGraphEdgesMap[a_id];
  auto aDims = Dimensions(aEdge);
  assert(mGraphEdgesMap.find(b_id) != mGraphEdgesMap.end());
  auto bEdge = mGraphEdgesMap[b_id];
  auto bDims = Dimensions(bEdge);
  auto outputDims = ConvertDimensions(output_desc->dimensions);
  std::vector<std::shared_ptr<EdgeInfoBase>> inputEdges = {aEdge, bEdge};

  // The shape of a tensor is 2D definited in WebNN Spec, but DML only support
  // 4D, so expand dimensions to 4D.
  assert(aDims.size() == 2);
  aDims = ExpandDimensions(aDims, 4);
  std::shared_ptr<DmlTensorDesc> aDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, aDmlTensorDesc,
                           &aEdge->outputTensorDESC, aDims)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC aTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                 &aDmlTensorDesc->bufferDesc};

  assert(bDims.size() == 2);
  bDims = ExpandDimensions(bDims, 4);
  std::shared_ptr<DmlTensorDesc> bDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, bDmlTensorDesc,
                           &bEdge->outputTensorDESC, bDims)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC bTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                 &bDmlTensorDesc->bufferDesc};

  assert(outputDims.size() == 2);
  auto expandedOutputDims = ExpandDimensions(outputDims, 4);
  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &aEdge->outputTensorDESC, expandedOutputDims, {},
                           true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                      &outputDmlTensorDesc->bufferDesc};

  // The operand c is optional.
  DML_TENSOR_DESC* cTensorDescPtr = nullptr;
  DML_TENSOR_DESC cTensorDesc;
  if (options->c_id != 0) {
    assert(mGraphEdgesMap.find(options->c_id) != mGraphEdgesMap.end());
    auto cEdge = mGraphEdgesMap[options->c_id];
    auto cDims = Dimensions(cEdge);
    // It is either a scalar, or of the shape that is unidirectionally
    // broadcastable to the shape [M, N] definited in WebNN Spec, DML only
    // support 4D, so broadCast the Shape of optional C to {1, 1, M, N }
    // supported in DML.
    auto cNewDims = expandedOutputDims;
    auto cNewStrides =
        CalculateStridesForBroadcast(cDims, cNewDims, cEdge->outputTensorDESC);
    std::shared_ptr<DmlTensorDesc> cDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, cDmlTensorDesc,
                             &cEdge->outputTensorDESC, cNewDims, cNewStrides)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    cTensorDesc = {DML_TENSOR_TYPE_BUFFER, &cDmlTensorDesc->bufferDesc};
    cTensorDescPtr = &cTensorDesc;
    inputEdges.push_back(cEdge);
  }

  DML_MATRIX_TRANSFORM aTranspose = options->a_transpose
                                        ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                        : DML_MATRIX_TRANSFORM_NONE;
  DML_MATRIX_TRANSFORM bTranspose = options->b_transpose
                                        ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                        : DML_MATRIX_TRANSFORM_NONE;
  DML_GEMM_OPERATOR_DESC gemm_desc = {};
  gemm_desc.ATensor = &aTensorDesc;
  gemm_desc.BTensor = &bTensorDesc;
  gemm_desc.CTensor = cTensorDescPtr;
  gemm_desc.OutputTensor = &outputTensorDesc;
  gemm_desc.TransA = aTranspose;
  gemm_desc.TransB = bTranspose;
  gemm_desc.Alpha = options->alpha;
  gemm_desc.Beta = options->beta;
  DML_OPERATOR_DESC dmlOperatorDesc = {};
  dmlOperatorDesc.Type = DML_OPERATOR_GEMM;
  dmlOperatorDesc.Desc = &gemm_desc;

  ComPtr<IDMLOperator> dmlOperator;
  WEBNN_CHECK(
      mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
  mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

  mGraphEdgesMap[output_desc->object_id] =
      CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
  AddEdgesToThisNode({inputEdges});

  // Reshape back according to output rank if needed to update the output edge.
  if (outputDims.size() < expandedOutputDims.size()) {
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                             &aEdge->outputTensorDESC, outputDims, {}, true)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
  }
  return;
}

void GraphDMLImpl::AddPool2d(uint32_t input_id,
                             Pool2dOptionsPtr options,
                             Pool2dType type,
                             OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(mGraphEdgesMap.find(input_id) != mGraphEdgesMap.end());

  auto inputEdge = mGraphEdgesMap[input_id];
  auto inputDims = Dimensions(inputEdge);
  auto outputDims = ConvertDimensions(output_desc->dimensions);
  std::vector<UINT> newInputDims = inputDims, newOutputDims = outputDims,
                    newInputStrides;

  DML_TENSOR_DESC inputTensorDesc = inputEdge->outputTensorDESC;
  if (options->layout == InputOperandLayout::kNhwc) {
    newInputDims = transposeDimensions(NhwcToNchw, inputDims);
    newOutputDims = transposeDimensions(NhwcToNchw, outputDims);
    newInputStrides = transposeStridesToNchw(inputDims, inputTensorDesc);

    std::shared_ptr<DmlTensorDesc> inputDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, inputDmlTensorDesc,
                             &inputEdge->outputTensorDESC, newInputDims,
                             newInputStrides)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    inputTensorDesc = {DML_TENSOR_TYPE_BUFFER, &inputDmlTensorDesc->bufferDesc};
  }

  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &inputEdge->outputTensorDESC, newOutputDims, {},
                           true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                      &outputDmlTensorDesc->bufferDesc};

  std::vector<UINT> strides = ConvertDimensions(options->strides);
  std::vector<UINT> dilations = ConvertDimensions(options->dilations);

  std::vector<UINT> windowSizes;
  if (!options->window_dimensions.empty()) {
    windowSizes = ConvertDimensions(options->window_dimensions);
  } else {
    windowSizes = {newInputDims[2], newInputDims[3]};
  }

  // TODO:: Support AutoPad::kSameUpper and kSameLower;
  auto padding = options->auto_pad == AutoPad::kExplicit
                     ? ExplicitPadding<Pool2dOptions>(options.get())
                     : ImplicitPadding<Pool2dOptions>(
                           options.get(), newInputDims, windowSizes);
  std::vector<UINT> startPadding = {padding[0], padding[2]};
  std::vector<UINT> endPadding = {padding[1], padding[3]};

  ComPtr<IDMLOperator> dmlOperator;
  if (type == Pool2dType::kAveragePool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      DAWN_INTERNAL_ERROR("The dilations of average pool2d are not supported.");
    }
    DML_AVERAGE_POOLING_OPERATOR_DESC dml_desc = {};
    dml_desc.InputTensor = &inputTensorDesc;
    dml_desc.OutputTensor = &outputTensorDesc;
    dml_desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    dml_desc.Strides = strides.data();
    dml_desc.WindowSize = windowSizes.data();
    dml_desc.StartPadding = startPadding.data();
    dml_desc.EndPadding = endPadding.data();
    dml_desc.IncludePadding = false;
    DML_OPERATOR_DESC dmlOperatorDesc = {};
    dmlOperatorDesc.Type = DML_OPERATOR_AVERAGE_POOLING;
    dmlOperatorDesc.Desc = &dml_desc;
    WEBNN_CHECK(
        mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
  } else if (type == Pool2dType::kL2Pool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      DAWN_INTERNAL_ERROR("The dilations of L2 pool2d are not supported.");
    }
    DML_LP_POOLING_OPERATOR_DESC dml_desc = {};
    dml_desc.InputTensor = &inputTensorDesc;
    dml_desc.OutputTensor = &outputTensorDesc;
    dml_desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    dml_desc.Strides = strides.data();
    dml_desc.WindowSize = windowSizes.data();
    dml_desc.StartPadding = startPadding.data();
    dml_desc.EndPadding = endPadding.data();
    dml_desc.P = 2;
    DML_OPERATOR_DESC dmlOperatorDesc = {};
    dmlOperatorDesc.Type = DML_OPERATOR_LP_POOLING;
    dmlOperatorDesc.Desc = &dml_desc;
    WEBNN_CHECK(
        mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
  } else if (type == Pool2dType::kMaxPool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      for (size_t i = 0; i < windowSizes.size(); ++i) {
        uint32_t paddedInputSize =
            newInputDims[2 + i] + startPadding[i] + endPadding[i];
        uint32_t dilatedWindowSize = 1 + (windowSizes[i] - 1) * dilations[i];
        newOutputDims[2 + i] =
            (dilatedWindowSize >= paddedInputSize)
                ? 1
                : (paddedInputSize - dilatedWindowSize) / strides[i] + 1;
      }
      outputDims = transposeDimensions(NchwToNhwc, newOutputDims);
      // Update output tensor.
      if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                               newOutputDims)) {
        DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
      }
    }

    DML_MAX_POOLING2_OPERATOR_DESC desc = {};
    desc.InputTensor = &inputTensorDesc;
    desc.OutputTensor = &outputTensorDesc;
    desc.OutputIndicesTensor = nullptr;
    desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    desc.Strides = strides.data();
    desc.WindowSize = windowSizes.data();
    desc.StartPadding = startPadding.data();
    desc.EndPadding = endPadding.data();
    desc.Dilations = dilations.data();
    DML_OPERATOR_DESC dmlOperatorDesc = {};
    dmlOperatorDesc.Type = DML_OPERATOR_MAX_POOLING2;
    dmlOperatorDesc.Desc = &desc;
    WEBNN_CHECK(
        mDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(&dmlOperator)));
  } else {
    DAWN_INTERNAL_ERROR("This pool2d type is not supported.");
  }
  mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;

  auto outputEdge =
      CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
  AddEdgesToThisNode({inputEdge});

  // Transpose output from nchw->nhwc.
  if (options->layout == InputOperandLayout::kNhwc) {
    TransposeOutputToNhwc(outputEdge, newOutputDims);
  }

  mGraphEdgesMap[output_desc->object_id] = outputEdge;
  return;
}

void GraphDMLImpl::AddUnary(uint32_t input_id,
                            UnaryOperandType type,
                            OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(mGraphEdgesMap.find(input_id) != mGraphEdgesMap.end());

  auto inputEdge = mGraphEdgesMap[input_id];
  auto inputDims = Dimensions(inputEdge);
  std::vector<std::shared_ptr<EdgeInfoBase>> inputEdges = {inputEdge};
  DML_TENSOR_DESC inputTensorDesc = inputEdge->outputTensorDESC;
  ComPtr<IDMLOperator> dmlOperator;
  switch (type) {
    case UnaryOperandType::kRelu: {
      CREATE_UNARY_OPERATOR(ACTIVATION_RELU, inputTensorDesc, dmlOperator);
    } break;
    case UnaryOperandType::kSigmoid: {
      CREATE_UNARY_OPERATOR(ACTIVATION_SIGMOID, inputTensorDesc, dmlOperator);
    } break;
    case UnaryOperandType::kSoftmax: {
      CREATE_UNARY_OPERATOR(ACTIVATION_SOFTMAX, inputTensorDesc, dmlOperator);
    } break;
    default:
      assert(0);
      break;
  }
  mIntermediateNodesMap[mIntermediateNodes.size()] = dmlOperator;
  mGraphEdgesMap[output_desc->object_id] =
      CreateEdgeFromThisNode(inputTensorDesc, mIntermediateNodes.size());
  AddEdgesToThisNode(inputEdges);
  return;
}

void GraphDMLImpl::AddFusionClamp(ClampOptionsPtr options,
                                  uint32_t operator_id) {
  fusion_operators_->AddClampOption(operator_id, std::move(options));
}

void GraphDMLImpl::AddOutput(const std::string& name, uint32_t operand_id) {
  assert(mGraphEdgesMap.find(operand_id) != mGraphEdgesMap.end());
  auto outputEdge = mGraphEdgesMap[operand_id];
  assert(outputEdge != nullptr);

  const DML_BUFFER_TENSOR_DESC* bufferDesc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
          outputEdge->outputTensorDESC.Desc);
  assert(bufferDesc != nullptr);
  auto* strides = bufferDesc->Strides;

  // Append identity to avoid directly using graph input as output, and
  // avoid lack of considering the impacts of strides if there are.
  if (outputEdge->isInputEdge || strides != nullptr) {
    auto edge = outputEdge;
    auto inputTensorDesc = outputEdge->outputTensorDESC;

    std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                             &inputTensorDesc)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                        &outputDmlTensorDesc->bufferDesc};

    ComPtr<IDMLOperator> dmlOperator;
    APPEND_IDENTITY(inputTensorDesc, outputTensorDesc, dmlOperator);
    outputEdge =
        CreateEdgeFromThisNode(outputTensorDesc, mIntermediateNodes.size());
    AddEdgesToThisNode({edge});
  }
  outputEdge->name = name;
  std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdgeDesc(
      new DML_OUTPUT_GRAPH_EDGE_DESC);
  auto* outputEdgeInfo = reinterpret_cast<EdgeInfo*>(outputEdge.get());
  outputEdgeDesc->FromNodeIndex = outputEdgeInfo->nodeIndex;
  outputEdgeDesc->FromNodeOutputIndex = outputEdgeInfo->outputNodeIndex;
  outputEdgeDesc->GraphOutputIndex = mOutputs.size();
  mOutputEdges.push_back({DML_GRAPH_EDGE_TYPE_OUTPUT, outputEdgeDesc.get()});
  mOutputEdgesDesc.push_back(std::move(outputEdgeDesc));

  mOutputs.push_back(*outputEdgeInfo);
  return;
}

BuildResult GraphDMLImpl::Finish() {
  WEBNN_CHECK(mDevice.Get()->QueryInterface(IID_PPV_ARGS(&mDevice1)));

  // Compiles a graph of DirectML operators into an object that can be
  // dispatched to the GPU.
  DML_GRAPH_DESC graphDesc = {};
  graphDesc.InputCount = static_cast<UINT>(mInputs.size());
  graphDesc.OutputCount = static_cast<UINT>(mOutputs.size());
  graphDesc.NodeCount = static_cast<UINT>(mIntermediateNodes.size());
  graphDesc.Nodes = mIntermediateNodes.data();
  graphDesc.InputEdgeCount = static_cast<UINT>(mInputEdges.size());
  graphDesc.InputEdges = mInputEdges.data();
  graphDesc.OutputEdgeCount = static_cast<UINT>(mOutputEdges.size());
  graphDesc.OutputEdges = mOutputEdges.data();
  graphDesc.IntermediateEdgeCount =
      static_cast<UINT>(mIntermediateEdges.size());
  graphDesc.IntermediateEdges = mIntermediateEdges.data();

  WEBNN_CHECK(mDevice1->CompileGraph(&graphDesc, DML_EXECUTION_FLAG_NONE,
                                     IID_PPV_ARGS(&mCompiledOperator)));
  return BuildResult::kOk;
}

void GraphDMLImpl::Build(
    const base::flat_map<std::string, uint32_t>& named_operands,
    ConstantsInfoPtr constants_info,
    BuildCallback callback) {
  // Add Output with named operands.
  for (auto& [name, operand_id] : named_operands) {
    AddOutput(name, operand_id);
  }

  // Finish the graph build.
  Finish();

  // Upload the data to GPU so that the constant data are not saved as member
  // variable.
  std::unique_ptr<UploadHeap> uploader =
      std::make_unique<UploadHeap>(execution_context_.get());
  ComPtr<ID3D12Resource> constants_resource = nullptr;
  if (constants_info.get() != nullptr) {
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        constants_info->shared_memory;
    size_t constants_byte_length = shared_memory_region.GetSize();
    UnorderedResources* unordered_resources =
        execution_context_->GetUnorderedResources();
    constants_resource = unordered_resources->Allocate(constants_byte_length);
    uploader->UploadConstants(constants_resource.Get(), constants_info);
  }
  std::vector<DML_BUFFER_BINDING> input_buffer_binding(mInputs.size());
  for (size_t i = 0; i < mInputs.size(); ++i) {
    auto input = mInputs[i];
    if (input->isConstantInput) {
      input_buffer_binding[i].Buffer = constants_resource.Get();
      auto& memory_info = constants_info->constants[input->object_id];
      input_buffer_binding[i].Offset = memory_info->byte_offset;
      input_buffer_binding[i].SizeInBytes = memory_info->byte_length;
    }
  }

  DML_BUFFER_ARRAY_BINDING input_buffer_array_binding = {};
  input_buffer_array_binding.BindingCount = input_buffer_binding.size();
  input_buffer_array_binding.Bindings = input_buffer_binding.data();
  DML_BINDING_DESC input_binding_desc{DML_BINDING_TYPE_BUFFER_ARRAY,
                                      &input_buffer_array_binding};

  execution_context_->InitializeGraph(graph_id_, mCompiledOperator.Get(),
                                      input_binding_desc);

  CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator,
                        mD3D12Device);

  for (auto& input : mInputs) {
    if (!input->isConstantInput) {
      uint64_t offset =
          RoundUpToMultiple(mCommonInputsResourceSize,
                            (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
      mCommonInputsResourceSize = offset + input->byteLength;
    }
  }
  if (mCommonInputsResourceSize) {
    D3D12_HEAP_PROPERTIES upload_heap_properties =
        CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC upload_resource_desc =
        CreateResourceDesc(mCommonInputsResourceSize);
    WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
        &upload_heap_properties, D3D12_HEAP_FLAG_NONE, &upload_resource_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&mUploadResource)));

    D3D12_HEAP_PROPERTIES input_heap_properties = CreateHeapProperties();
    D3D12_RESOURCE_DESC input_resource_desc = CreateResourceDesc(
        mCommonInputsResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
        &input_heap_properties, D3D12_HEAP_FLAG_NONE, &input_resource_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&mInputResource)));
  }

  for (size_t i = 0; i < mOutputs.size(); ++i) {
    uint64_t byteLength = reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
                              mOutputs[i].outputTensorDESC.Desc)
                              ->TotalTensorSizeInBytes;
    uint64_t offset = RoundUpToMultiple(
        mOutputsResourceSize, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
    MemoryInfo memory_info = {};
    memory_info.byte_offset = offset;
    memory_info.byte_length = byteLength;

    outputs_info_map_[mOutputs[i].name] = memory_info;
    mOutputsResourceSize = offset + byteLength;
  }
  outputs_shm_region_ =
      base::ReadOnlySharedMemoryRegion::Create(mOutputsResourceSize);
  // for (auto& output : mOutputs) {
  //   MemoryInfo memory_info = outputs_info_map_[output.name];
  //   outputs_shm_mapping_[output.name] = outputs_shm_region_.MapAt(
  //       memory_info.byte_offset, memory_info.byte_length);
  // }

  if (mOutputsResourceSize) {
    D3D12_HEAP_PROPERTIES output_heap_properties = CreateHeapProperties();
    D3D12_RESOURCE_DESC output_resource_desc = CreateResourceDesc(
        mOutputsResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    WEBNN_CHECK(mD3D12Device->CreateCommittedResource(
        &output_heap_properties, D3D12_HEAP_FLAG_NONE, &output_resource_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&mOutputResource)));

    D3D12_HEAP_PROPERTIES readback_heap_properties =
        CreateHeapProperties(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC readback_resource_desc =
        CreateResourceDesc(mOutputsResourceSize);
    mD3D12Device->CreateCommittedResource(
        &readback_heap_properties, D3D12_HEAP_FLAG_NONE,
        &readback_resource_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&mReadBackResource));
  }
  std::move(callback).Run(BuildResult::kOk);
}

void GraphDMLImpl::Compute(NamedInputsPtr named_inputs,
                           ComputeCallback callback) {
  auto named_outputs = ml::webnn::mojom::NamedOutputs::New();

  std::vector<DML_BINDING_DESC> outputBindingDesc(mOutputs.size());
  // The sort of the outputs from Graph Compute is different from the
  // outputs from Graph Build, so the offset need to be found the corrent output
  // with name to read back from GPU buffer.
  base::flat_map<std::string, DML_BUFFER_BINDING> output_buffer_binding;
  uint64_t outputOffset = 0;
  for (size_t i = 0; i < mOutputs.size(); ++i) {
    size_t byteLength = reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
                            mOutputs[i].outputTensorDESC.Desc)
                            ->TotalTensorSizeInBytes;
    outputOffset = RoundUpToMultiple(
        outputOffset, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
    DML_BUFFER_BINDING buffer_binding;
    buffer_binding.Buffer = mOutputResource.Get();
    buffer_binding.Offset = outputOffset;
    buffer_binding.SizeInBytes = byteLength;
    std::string name = mOutputs[i].name;
    output_buffer_binding[name] = buffer_binding;
    outputBindingDesc[i] = {DML_BINDING_TYPE_BUFFER,
                            &output_buffer_binding[name]};
    outputOffset = outputOffset + byteLength;
  }

  execution_context_->ExecuteGraph(graph_id_, mCompiledOperator.Get(),
                                   std::move(named_inputs), mInputs,
                                   outputBindingDesc, mCommonInputsResourceSize,
                                   mUploadResource, mInputResource);

  // Copy buffer from outputResource to readBackResource.
  CopyBufferRegionUtil(mCommandList, mOutputResource, mReadBackResource,
                       mOutputsResourceSize, D3D12_RESOURCE_STATE_COPY_SOURCE,
                       false);

  CloseExecuteResetWait(mCommandList, mCommandQueue, mCommandAllocator,
                        mD3D12Device);

  D3D12_RANGE tensorBufferRange{0, mOutputsResourceSize};
  int8_t* readBackBuffer;
  WEBNN_CHECK(mReadBackResource->Map(
      0, &tensorBufferRange, reinterpret_cast<void**>(&readBackBuffer)));

  for (auto& [name, memory_info] : outputs_info_map_) {
    auto mojo_memory_info = ml::webnn::mojom::MemoryInfo::New();
    mojo_memory_info->byte_offset = memory_info.byte_offset;
    mojo_memory_info->byte_length = memory_info.byte_length;
    named_outputs->outputs[name] = std::move(mojo_memory_info);

    DML_BUFFER_BINDING buffer_binding = output_buffer_binding[name];
    std::vector<uint8_t> output_buffer(buffer_binding.SizeInBytes);
    uint8_t* address = outputs_shm_region_.mapping.GetMemoryAs<uint8_t>() +
                       memory_info.byte_offset;
    memcpy(address, readBackBuffer + buffer_binding.Offset,
           buffer_binding.SizeInBytes);
  }
  named_outputs->shared_memory = outputs_shm_region_.region.Duplicate();

  mReadBackResource->Unmap(0, nullptr);

  std::move(callback).Run(ComputeResult::kOk, std::move(named_outputs));
}

}  // namespace content::webnn

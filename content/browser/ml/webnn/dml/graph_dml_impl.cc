// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/graph_dml_impl.h"

#include "base/logging.h"
#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/execution_context.h"
#include "content/browser/ml/webnn/dml/execution_resources.h"
#include "content/browser/ml/webnn/dml/graph_dml_impl.h"
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
    const DML_TENSOR_DESC& input_tensor_desc) {
  const DML_BUFFER_TENSOR_DESC* bufferDesc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(input_tensor_desc.Desc);
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
    const DML_TENSOR_DESC& input_tensor_desc,
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
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(input_tensor_desc.Desc);
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

std::vector<UINT> Dimensions(NodeOutput* node_output) {
  DML_TENSOR_DESC& tensor_desc = node_output->GetTensorDesc();
  const DML_BUFFER_TENSOR_DESC* desc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(tensor_desc.Desc);

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

}  // namespace

DmlTensorDesc::DmlTensorDesc() = default;
DmlTensorDesc::~DmlTensorDesc() = default;

InputEdgeInfo::InputEdgeInfo() = default;
InputEdgeInfo::~InputEdgeInfo() = default;

#define DAWN_INTERNAL_ERROR(MESSAGE)            \
  do {                                          \
    error_messages_ = MESSAGE;                  \
    assert(0);                                  \
    build_result_ = BuildResult::kUnknownError; \
    return;                                     \
  } while (0)

#define CREATE_BINARY_OPERATOR(type, a_tensor_desc, b_tensor_desc, \
                               output_tensor, node)                \
  DML_ELEMENT_WISE_##type##_OPERATOR_DESC operator_desc{};         \
  operator_desc.ATensor = &a_tensor_desc;                          \
  operator_desc.BTensor = &b_tensor_desc;                          \
  operator_desc.OutputTensor = &output_tensor;                     \
  node = graph_desc_builder_->CreateOperatorNode(                  \
      DML_OPERATOR_ELEMENT_WISE_##type, &operator_desc);

#define CREATE_UNARY_OPERATOR(type, input_tensor_desc, dmlOperator)   \
  DML_##type##_OPERATOR_DESC operator_desc{};                         \
  operator_desc.InputTensor = &input_tensor_desc;                     \
  operator_desc.OutputTensor = &input_tensor_desc;                    \
  node = graph_desc_builder_->CreateOperatorNode(DML_OPERATOR_##type, \
                                                 &operator_desc);

// Append IDENTITY to remove the strides of input tensor. Use this to implement
// Reshape, Squeeze, Transpose and avoid creating an invaild graph with input =
// output.
#define APPEND_IDENTITY(input_tensor_desc, output_tensor, node) \
  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC operator_desc{};      \
  operator_desc.InputTensor = &input_tensor_desc;               \
  operator_desc.OutputTensor = &output_tensor;                  \
  node = graph_desc_builder_->CreateOperatorNode(               \
      DML_OPERATOR_ELEMENT_WISE_IDENTITY, &operator_desc);

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
    : graph_id_(graph_id),
      execution_context_(execution_context),
      input_resource_uploader_(
          std::make_unique<UploadHeap>(execution_context_.get())),
      output_resource_readback_(
          std::make_unique<ReadbackHeap>(execution_context_.get())),
      graph_desc_builder_(std::make_unique<GraphDescBuilder>(
          execution_context->GetDMLDevice())),
      fusion_operators_(std::make_unique<FusionOperators>()) {}

void GraphDMLImpl::AddInput(const std::string& name,
                            OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  // const OperandDescriptor* desc = input->GetOperandDescriptor();
  std::shared_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, dmlTensorDesc, desc)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                 &(dmlTensorDesc->bufferDesc)};
  Node input_node = graph_desc_builder_->CreateInputNode(std::move(name));
  auto node_output =
      graph_desc_builder_->CreateNodeOutput(input_node, 0, tensor_desc);
  node_output_map_[desc->object_id] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddConstant(OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  if (node_output_map_.find(desc->object_id) != node_output_map_.end()) {
    LOG(ERROR) << "There are issues in sorting graph";
    return;
  }
  std::shared_ptr<DmlTensorDesc> dmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, dmlTensorDesc, desc,
                           DML_TENSOR_FLAG_OWNED_BY_DML)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                 &(dmlTensorDesc->bufferDesc)};
  // std::string name = "Input_Constant_" + std::to_string(mInputs.size());
  Node constant_node = graph_desc_builder_->CreateConstantNode(desc->object_id);
  auto node_output =
      graph_desc_builder_->CreateNodeOutput(constant_node, 0, tensor_desc);
  node_output_map_[desc->object_id] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddElementWiseBinary(uint32_t a_id,
                                        uint32_t b_id,
                                        BinaryOperandType type,
                                        OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(node_output_map_.find(a_id) != node_output_map_.end());
  assert(node_output_map_.find(b_id) != node_output_map_.end());

  auto* a_node_output = node_output_map_[a_id].get();
  auto* b_node_output = node_output_map_[b_id].get();
  auto a_dims = Dimensions(a_node_output);
  auto b_dims = Dimensions(b_node_output);
  auto output_dims = ConvertDimensions(output_desc->dimensions);
  size_t broadcastSkipAxis = 0;
  std::vector<UINT> a_new_dims = output_dims, b_new_dims = output_dims,
                    output_new_dims = output_dims;

  // TODO:: Remove the broadcast which is done in blink side.
  auto a_new_strides = CalculateStridesForBroadcast(
      a_dims, a_new_dims, a_node_output->GetTensorDesc(), broadcastSkipAxis);
  std::shared_ptr<DmlTensorDesc> aDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, aDmlTensorDesc,
                           &a_node_output->GetTensorDesc(), a_new_dims,
                           a_new_strides)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC a_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                   &aDmlTensorDesc->bufferDesc};

  auto b_new_strides = CalculateStridesForBroadcast(
      b_dims, b_new_dims, b_node_output->GetTensorDesc(), broadcastSkipAxis);
  std::shared_ptr<DmlTensorDesc> bDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, bDmlTensorDesc,
                           &b_node_output->GetTensorDesc(), b_new_dims,
                           b_new_strides)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC b_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                   &bDmlTensorDesc->bufferDesc};

  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &a_node_output->GetTensorDesc(), output_new_dims, {},
                           true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC output_tensor = {DML_TENSOR_TYPE_BUFFER,
                                   &outputDmlTensorDesc->bufferDesc};
  Node node;
  switch (type) {
    case BinaryOperandType::kAdd: {
      CREATE_BINARY_OPERATOR(ADD, a_tensor_desc, b_tensor_desc, output_tensor,
                             node);
    } break;
    case BinaryOperandType::kDiv: {
      CREATE_BINARY_OPERATOR(DIVIDE, a_tensor_desc, b_tensor_desc,
                             output_tensor, node);
    } break;
    case BinaryOperandType::kMul: {
      CREATE_BINARY_OPERATOR(MULTIPLY, a_tensor_desc, b_tensor_desc,
                             output_tensor, node);
    } break;
    case BinaryOperandType::kSub: {
      CREATE_BINARY_OPERATOR(SUBTRACT, a_tensor_desc, b_tensor_desc,
                             output_tensor, node);
    } break;
    case BinaryOperandType::kMax: {
      CREATE_BINARY_OPERATOR(MAX, a_tensor_desc, b_tensor_desc, output_tensor,
                             node);
    } break;
    case BinaryOperandType::kMin: {
      CREATE_BINARY_OPERATOR(MIN, a_tensor_desc, b_tensor_desc, output_tensor,
                             node);
    } break;
    default:
      DAWN_INTERNAL_ERROR(" Binary op is not implemented.");
  }
  if (output_dims != output_new_dims) {
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                             &a_node_output->GetTensorDesc(), output_dims, {},
                             true)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
  }

  graph_desc_builder_->Connect({a_node_output, b_node_output}, {node});
  auto node_output =
      graph_desc_builder_->CreateNodeOutput(node, 0, output_tensor);
  node_output_map_[output_desc->object_id] = std::move(node_output);
  return;
}

std::unique_ptr<NodeOutput> GraphDMLImpl::Clamp(NodeOutput* input_node,
                                                const ClampOptions* options) {
  DML_TENSOR_DESC input_tensor = input_node->GetTensorDesc();
  DML_ELEMENT_WISE_CLIP_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = &input_tensor;
  operator_desc.OutputTensor = &input_tensor;
  operator_desc.ScaleBias = nullptr;
  operator_desc.Min = options->minValue;
  operator_desc.Max = options->maxValue;
  Node operator_node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_ELEMENT_WISE_CLIP, &operator_desc);

  graph_desc_builder_->Connect({input_node}, {operator_node});
  return graph_desc_builder_->CreateNodeOutput(operator_node, 0, input_tensor);
}

void GraphDMLImpl::AddClamp(uint32_t input_id,
                            ClampOptionsPtr options,
                            OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  auto* input_node = node_output_map_[input_id].get();
  node_output_map_[output_desc->object_id] = Clamp(input_node, options.get());
  return;
}

void GraphDMLImpl::EmulateFusedOperator(const FusionOperator* activation,
                                        std::unique_ptr<NodeOutput>& input_node,
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
    input_node = Clamp(input_node.get(), options);
  }
  return;
}

void GraphDMLImpl::TransposeOutputToNhwc(
    std::unique_ptr<NodeOutput>& input_node,
    const std::vector<UINT>& nchwOutputDims) {
  auto nhwcOutputStrides = transposeStrides(NchwToNhwc, nchwOutputDims);
  auto nhwcOutputDims = transposeDimensions(NchwToNhwc, nchwOutputDims);
  std::shared_ptr<DmlTensorDesc> nhwcOutputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, nhwcOutputDmlTensorDesc,
                           &input_node->GetTensorDesc(), nhwcOutputDims,
                           nhwcOutputStrides, true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC nhwcOutputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                          &nhwcOutputDmlTensorDesc->bufferDesc};
  auto node = input_node->GetNode();
  input_node =
      graph_desc_builder_->CreateNodeOutput(node, 0, nhwcOutputTensorDesc);
  return;
}

void GraphDMLImpl::AddConv2d(uint32_t input_id,
                             uint32_t filter_id,
                             Conv2dOptionsPtr options,
                             OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(node_output_map_.find(input_id) != node_output_map_.end());
  assert(node_output_map_.find(filter_id) != node_output_map_.end());

  auto* input_node = node_output_map_[input_id].get();
  auto* filter_node = node_output_map_[filter_id].get();

  auto inputDims = Dimensions(input_node);
  auto filterDims = Dimensions(filter_node);
  auto output_dims = ConvertDimensions(output_desc->dimensions);
  std::vector<UINT> newInputDims = inputDims, newFilterDims = filterDims,
                    newOutputDims = output_dims, newInputStrides,
                    newFilterStrides;

  DML_TENSOR_DESC input_tensor_desc = input_node->GetTensorDesc();
  if (options->inputLayout == InputOperandLayout::kNhwc) {
    newInputDims = transposeDimensions(NhwcToNchw, inputDims);
    newOutputDims = transposeDimensions(NhwcToNchw, output_dims);
    newInputStrides = transposeStridesToNchw(inputDims, input_tensor_desc);

    std::shared_ptr<DmlTensorDesc> inputDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, inputDmlTensorDesc,
                             &input_node->GetTensorDesc(), newInputDims,
                             newInputStrides)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    input_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                         &inputDmlTensorDesc->bufferDesc};
  }

  DML_TENSOR_DESC filterTensorDesc = filter_node->GetTensorDesc();
  if (options->filterLayout != Conv2dFilterOperandLayout::kOihw) {
    newFilterDims =
        transposeFilterDimensionsAsOihw(options->filterLayout, filterDims);
    newFilterStrides =
        transposeFilterStridesAsOihw(options->filterLayout, filterDims);

    std::shared_ptr<DmlTensorDesc> filterDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, filterDmlTensorDesc,
                             &filter_node->GetTensorDesc(), newFilterDims,
                             newFilterStrides)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    filterTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                        &filterDmlTensorDesc->bufferDesc};
  }

  std::vector<NodeOutput*> input_nodes = {input_node, filter_node};
  const DML_TENSOR_DESC* biasTensorDescPtr = nullptr;
  DML_TENSOR_DESC newBiasTensorDesc = {};
  if (options->bias_id != 0) {
    assert(node_output_map_.find(options->bias_id) != node_output_map_.end());
    auto* bias_node = node_output_map_[options->bias_id].get();
    auto biasDims = Dimensions(bias_node);
    if (biasDims[0] != newFilterDims[0] || biasDims.size() != 1) {
      DAWN_INTERNAL_ERROR(
          "The bias should be 1-D tensor with the shape of [output_channels].");
    }

    // Reshape bias from 1-D to 4-D for NCHW layout.
    std::vector<UINT> newBiasDims = {1, biasDims[0], 1, 1};
    std::shared_ptr<DmlTensorDesc> biasDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, biasDmlTensorDesc,
                             &bias_node->GetTensorDesc(), newBiasDims)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    newBiasTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                         &biasDmlTensorDesc->bufferDesc};
    biasTensorDescPtr = &newBiasTensorDesc;
    input_nodes.push_back(bias_node);
  }

  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &input_node->GetTensorDesc(), newOutputDims, {},
                           true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC output_tensor = {DML_TENSOR_TYPE_BUFFER,
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
  DML_CONVOLUTION_OPERATOR_DESC operator_desc{};
  operator_desc.InputTensor = &input_tensor_desc;
  operator_desc.FilterTensor = &filterTensorDesc;
  operator_desc.BiasTensor = biasTensorDescPtr;
  operator_desc.OutputTensor = &output_tensor;

  operator_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
  operator_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
  operator_desc.DimensionCount = inputDims.size() - 2;
  operator_desc.Strides = strides.data();
  operator_desc.Dilations = dilations.data();
  operator_desc.StartPadding = startPadding.data();
  operator_desc.EndPadding = endPadding.data();
  operator_desc.OutputPadding = defaultOutPadding.data();
  operator_desc.GroupCount = static_cast<UINT>(options->groups);
  operator_desc.FusedActivation = fusedActivation;

  Node operator_node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_CONVOLUTION, &operator_desc);
  graph_desc_builder_->Connect(std::move(input_nodes), operator_node);
  auto output_node =
      graph_desc_builder_->CreateNodeOutput(operator_node, 0, output_tensor);

  // Transpose output from nchw->nhwc.
  if (options->inputLayout == InputOperandLayout::kNhwc) {
    TransposeOutputToNhwc(output_node, newOutputDims);
  }

  EmulateFusedOperator(options->activation.get(), output_node, output_dims);
  node_output_map_[output_desc->object_id] = std::move(output_node);
  return;
}

void GraphDMLImpl::AddReshape(uint32_t input_id,
                              OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(node_output_map_.find(input_id) != node_output_map_.end());

  auto* input_node = node_output_map_[input_id].get();
  auto output_dims = ConvertDimensions(output_desc->dimensions);
  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  // Reshape needn't new strides, because the layout has not been changed.
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &input_node->GetTensorDesc(), output_dims)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC output_tensor = {DML_TENSOR_TYPE_BUFFER,
                                   &outputDmlTensorDesc->bufferDesc};
  // Reshape is not a real node in DML, just need to update node output with new
  // tensor.
  auto node = input_node->GetNode();
  node_output_map_[output_desc->object_id] =
      graph_desc_builder_->CreateNodeOutput(node, 0, output_tensor);
  return;
}

void GraphDMLImpl::AddGemm(uint32_t a_id,
                           uint32_t b_id,
                           GemmOptionsPtr options,
                           OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(node_output_map_.find(a_id) != node_output_map_.end());
  auto* a_node_output = node_output_map_[a_id].get();
  auto a_dims = Dimensions(a_node_output);
  assert(node_output_map_.find(b_id) != node_output_map_.end());
  auto* b_node_output = node_output_map_[b_id].get();
  auto b_dims = Dimensions(b_node_output);
  auto output_dims = ConvertDimensions(output_desc->dimensions);

  // The shape of a tensor is 2D definited in WebNN Spec, but DML only support
  // 4D, so expand dimensions to 4D.
  assert(a_dims.size() == 2);
  a_dims = ExpandDimensions(a_dims, 4);
  std::shared_ptr<DmlTensorDesc> aDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, aDmlTensorDesc,
                           &a_node_output->GetTensorDesc(), a_dims)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC a_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                   &aDmlTensorDesc->bufferDesc};

  assert(b_dims.size() == 2);
  b_dims = ExpandDimensions(b_dims, 4);
  std::shared_ptr<DmlTensorDesc> bDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, bDmlTensorDesc,
                           &b_node_output->GetTensorDesc(), b_dims)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC b_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                   &bDmlTensorDesc->bufferDesc};

  assert(output_dims.size() == 2);
  auto expandedOutputDims = ExpandDimensions(output_dims, 4);
  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &a_node_output->GetTensorDesc(), expandedOutputDims,
                           {}, true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC output_tensor = {DML_TENSOR_TYPE_BUFFER,
                                   &outputDmlTensorDesc->bufferDesc};

  // The operand c is optional.
  DML_TENSOR_DESC* cTensorDescPtr = nullptr;
  DML_TENSOR_DESC cTensorDesc;
  std::vector<NodeOutput*> input_nodes = {a_node_output, b_node_output};
  if (options->c_id != 0) {
    assert(node_output_map_.find(options->c_id) != node_output_map_.end());
    auto* c_node_output = node_output_map_[options->c_id].get();
    auto cDims = Dimensions(c_node_output);
    // It is either a scalar, or of the shape that is unidirectionally
    // broadcastable to the shape [M, N] definited in WebNN Spec, DML only
    // support 4D, so broadCast the Shape of optional C to {1, 1, M, N }
    // supported in DML.
    auto cNewDims = expandedOutputDims;
    auto cNewStrides = CalculateStridesForBroadcast(
        cDims, cNewDims, c_node_output->GetTensorDesc());
    std::shared_ptr<DmlTensorDesc> cDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, cDmlTensorDesc,
                             &c_node_output->GetTensorDesc(), cNewDims,
                             cNewStrides)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    cTensorDesc = {DML_TENSOR_TYPE_BUFFER, &cDmlTensorDesc->bufferDesc};
    cTensorDescPtr = &cTensorDesc;
    input_nodes.push_back(c_node_output);
  }

  DML_MATRIX_TRANSFORM aTranspose = options->a_transpose
                                        ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                        : DML_MATRIX_TRANSFORM_NONE;
  DML_MATRIX_TRANSFORM bTranspose = options->b_transpose
                                        ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                        : DML_MATRIX_TRANSFORM_NONE;
  DML_GEMM_OPERATOR_DESC gemm_desc = {};
  gemm_desc.ATensor = &a_tensor_desc;
  gemm_desc.BTensor = &b_tensor_desc;
  gemm_desc.CTensor = cTensorDescPtr;
  gemm_desc.OutputTensor = &output_tensor;
  gemm_desc.TransA = aTranspose;
  gemm_desc.TransB = bTranspose;
  gemm_desc.Alpha = options->alpha;
  gemm_desc.Beta = options->beta;

  Node operator_node =
      graph_desc_builder_->CreateOperatorNode(DML_OPERATOR_GEMM, &gemm_desc);
  graph_desc_builder_->Connect(std::move(input_nodes), {operator_node});
  node_output_map_[output_desc->object_id] =
      graph_desc_builder_->CreateNodeOutput(operator_node, 0, output_tensor);

  // Reshape back according to output rank if needed to update the output edge.
  if (output_dims.size() < expandedOutputDims.size()) {
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                             &a_node_output->GetTensorDesc(), output_dims, {},
                             true)) {
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
  assert(node_output_map_.find(input_id) != node_output_map_.end());

  auto* input_node = node_output_map_[input_id].get();
  auto inputDims = Dimensions(input_node);
  auto output_dims = ConvertDimensions(output_desc->dimensions);
  std::vector<UINT> newInputDims = inputDims, newOutputDims = output_dims,
                    newInputStrides;

  DML_TENSOR_DESC input_tensor_desc = input_node->GetTensorDesc();
  if (options->layout == InputOperandLayout::kNhwc) {
    newInputDims = transposeDimensions(NhwcToNchw, inputDims);
    newOutputDims = transposeDimensions(NhwcToNchw, output_dims);
    newInputStrides = transposeStridesToNchw(inputDims, input_tensor_desc);

    std::shared_ptr<DmlTensorDesc> inputDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, inputDmlTensorDesc,
                             &input_node->GetTensorDesc(), newInputDims,
                             newInputStrides)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    input_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                         &inputDmlTensorDesc->bufferDesc};
  }

  std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
  if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                           &input_node->GetTensorDesc(), newOutputDims, {},
                           true)) {
    DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
  }
  DML_TENSOR_DESC output_tensor = {DML_TENSOR_TYPE_BUFFER,
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

  Node operator_node;
  if (type == Pool2dType::kAveragePool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      DAWN_INTERNAL_ERROR("The dilations of average pool2d are not supported.");
    }
    DML_AVERAGE_POOLING_OPERATOR_DESC dml_desc = {};
    dml_desc.InputTensor = &input_tensor_desc;
    dml_desc.OutputTensor = &output_tensor;
    dml_desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    dml_desc.Strides = strides.data();
    dml_desc.WindowSize = windowSizes.data();
    dml_desc.StartPadding = startPadding.data();
    dml_desc.EndPadding = endPadding.data();
    dml_desc.IncludePadding = false;
    operator_node = graph_desc_builder_->CreateOperatorNode(
        DML_OPERATOR_AVERAGE_POOLING, &dml_desc);
  } else if (type == Pool2dType::kL2Pool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      DAWN_INTERNAL_ERROR("The dilations of L2 pool2d are not supported.");
    }
    DML_LP_POOLING_OPERATOR_DESC dml_desc = {};
    dml_desc.InputTensor = &input_tensor_desc;
    dml_desc.OutputTensor = &output_tensor;
    dml_desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    dml_desc.Strides = strides.data();
    dml_desc.WindowSize = windowSizes.data();
    dml_desc.StartPadding = startPadding.data();
    dml_desc.EndPadding = endPadding.data();
    dml_desc.P = 2;
    operator_node = graph_desc_builder_->CreateOperatorNode(
        DML_OPERATOR_LP_POOLING, &dml_desc);
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
      output_dims = transposeDimensions(NchwToNhwc, newOutputDims);
      // Update output tensor.
      if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                               newOutputDims)) {
        DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
      }
    }

    DML_MAX_POOLING2_OPERATOR_DESC desc = {};
    desc.InputTensor = &input_tensor_desc;
    desc.OutputTensor = &output_tensor;
    desc.OutputIndicesTensor = nullptr;
    desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    desc.Strides = strides.data();
    desc.WindowSize = windowSizes.data();
    desc.StartPadding = startPadding.data();
    desc.EndPadding = endPadding.data();
    desc.Dilations = dilations.data();
    operator_node = graph_desc_builder_->CreateOperatorNode(
        DML_OPERATOR_MAX_POOLING2, &desc);
  } else {
    DAWN_INTERNAL_ERROR("This pool2d type is not supported.");
  }
  graph_desc_builder_->Connect({input_node}, operator_node);
  auto output_node =
      graph_desc_builder_->CreateNodeOutput(operator_node, 0, output_tensor);

  // Transpose output from nchw->nhwc.
  if (options->layout == InputOperandLayout::kNhwc) {
    TransposeOutputToNhwc(output_node, newOutputDims);
  }

  node_output_map_[output_desc->object_id] = std::move(output_node);

  return;
}

void GraphDMLImpl::AddUnary(uint32_t input_id,
                            UnaryOperandType type,
                            OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  assert(node_output_map_.find(input_id) != node_output_map_.end());

  auto* input_node = node_output_map_[input_id].get();
  auto inputDims = Dimensions(input_node);
  DML_TENSOR_DESC input_tensor_desc = input_node->GetTensorDesc();
  Node node;
  switch (type) {
    case UnaryOperandType::kRelu: {
      CREATE_UNARY_OPERATOR(ACTIVATION_RELU, input_tensor_desc, node);
    } break;
    case UnaryOperandType::kSigmoid: {
      CREATE_UNARY_OPERATOR(ACTIVATION_SIGMOID, input_tensor_desc, node);
    } break;
    case UnaryOperandType::kSoftmax: {
      CREATE_UNARY_OPERATOR(ACTIVATION_SOFTMAX, input_tensor_desc, node);
    } break;
    default:
      assert(0);
      break;
  }
  graph_desc_builder_->Connect({input_node}, {node});
  auto node_output =
      graph_desc_builder_->CreateNodeOutput(node, 0, input_tensor_desc);
  node_output_map_[output_desc->object_id] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddFusionClamp(ClampOptionsPtr options,
                                  uint32_t operator_id) {
  fusion_operators_->AddClampOption(operator_id, std::move(options));
}

void GraphDMLImpl::AddOutput(const std::string& name, uint32_t operand_id) {
  assert(node_output_map_.find(operand_id) != node_output_map_.end());
  auto* output_node = node_output_map_[operand_id].get();
  assert(output_node != nullptr);

  const DML_BUFFER_TENSOR_DESC* bufferDesc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
          output_node->GetTensorDesc().Desc);
  assert(bufferDesc != nullptr);
  auto* strides = bufferDesc->Strides;

  // Append identity to avoid directly using graph input as output, and
  // avoid lack of considering the impacts of strides if there are.
  auto node = output_node->GetNode();
  if (node.type == NodeType::kInput || node.type == NodeType::kConstant ||
      strides != nullptr) {
    auto input_tensor_desc = output_node->GetTensorDesc();

    std::shared_ptr<DmlTensorDesc> outputDmlTensorDesc(new DmlTensorDesc);
    if (!CreateDmlTensorDesc(mDmlTensorsDesc, outputDmlTensorDesc,
                             &input_tensor_desc)) {
      DAWN_INTERNAL_ERROR("Failed to create DML tensor description.");
    }
    DML_TENSOR_DESC output_tensor = {DML_TENSOR_TYPE_BUFFER,
                                     &outputDmlTensorDesc->bufferDesc};

    ComPtr<IDMLOperator> dmlOperator;
    APPEND_IDENTITY(input_tensor_desc, output_tensor, node);
    graph_desc_builder_->Connect({output_node}, {node});
    std::unique_ptr<NodeOutput> identity_output_node =
        graph_desc_builder_->CreateNodeOutput(node, 0, output_tensor);
    graph_desc_builder_->AddOutputEdge(identity_output_node.get(), name);
  } else {
    graph_desc_builder_->AddOutputEdge(output_node, name);
  }
  return;
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
  mCompiledOperator = graph_desc_builder_->Compile(DML_EXECUTION_FLAG_NONE);

  // Upload the data to GPU so that the constant data are not saved as member
  // variable.
  std::unique_ptr<UploadHeap> uploader =
      std::make_unique<UploadHeap>(execution_context_.get());
  ComPtr<ID3D12Resource> constants_resource = nullptr;
  if (constants_info.get() != nullptr) {
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        constants_info->shared_memory;
    size_t constants_byte_length = shared_memory_region.GetSize();
    ExecutionResources* execution_resources =
        execution_context_->GetExecutionResources();
    constants_resource = execution_resources->Allocate(constants_byte_length);
    uploader->UploadConstants(constants_resource.Get(), constants_info);
  }

  auto input_nodes = graph_desc_builder_->GetInputNodes();
  std::vector<DML_BUFFER_BINDING> input_buffer_binding(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto input = input_nodes[i];
    if (input.type == NodeType::kConstant) {
      input_buffer_binding[i].Buffer = constants_resource.Get();
      auto& memory_info = constants_info->constants[input.object_id];
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

  execution_context_->Flush();

  auto& named_outputs = graph_desc_builder_->GetNamedOutputs();
  HRESULT hr = output_resource_readback_->InitializeResource(named_outputs);
  if (FAILED(hr)) {
    std::move(callback).Run(BuildResult::kUnknownError);
    return;
  }

  std::move(callback).Run(BuildResult::kOk);
  return;
}

void GraphDMLImpl::Compute(NamedInputsPtr named_inputs,
                           ComputeCallback callback) {
  ExecutionResources* execution_resources =
      execution_context_->GetExecutionResources();
  ID3D12Resource* inputs_resource =
      execution_resources->GetResource(graph_id_, ResourceType::kInput);
  if (inputs_resource == nullptr) {
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        named_inputs->shared_memory;
    DCHECK(shared_memory_region.IsValid());
    size_t inputs_byte_length = shared_memory_region.GetSize();
    inputs_resource = execution_resources->Allocate(
        ResourceType::kInput, inputs_byte_length, graph_id_);
  }
  input_resource_uploader_->UploadInputs(inputs_resource, named_inputs);
  auto input_nodes = graph_desc_builder_->GetInputNodes();
  std::vector<DML_BUFFER_BINDING> input_buffer_binding(input_nodes.size());
  std::vector<DML_BINDING_DESC> input_binding_desc(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto input = input_nodes[i];
    if (input.type == NodeType::kInput) {
      input_buffer_binding[i].Buffer = inputs_resource;
      auto& memory_info = named_inputs->inputs[input.name];
      input_buffer_binding[i].Offset = memory_info->byte_offset;
      input_buffer_binding[i].SizeInBytes = memory_info->byte_length;

      input_binding_desc[i] = {DML_BINDING_TYPE_BUFFER,
                               &input_buffer_binding[i]};
    }
  }

  ID3D12Resource* outputs_resource =
      execution_resources->GetResource(graph_id_, ResourceType::kOutput);
  if (outputs_resource == nullptr) {
    size_t outputs_resource_size =
        output_resource_readback_->GetOutputsResourceSize();
    outputs_resource = execution_resources->Allocate(
        ResourceType::kOutput, outputs_resource_size, graph_id_);
  }
  auto& output_length_map = graph_desc_builder_->GetNamedOutputs();
  std::vector<DML_BINDING_DESC> output_binding_desc(output_length_map.size());
  // The sort of the outputs from Graph Compute is different from the
  // outputs from Graph Build, so the offset need to be found the corrent output
  // with name to read back from GPU buffer.
  base::flat_map<std::string, DML_BUFFER_BINDING> output_buffer_binding;
  uint64_t aligned_offset = 0;
  size_t i = 0;
  for (auto& [name, byte_length] : output_length_map) {
    DML_BUFFER_BINDING buffer_binding;
    buffer_binding.Buffer = outputs_resource;
    buffer_binding.Offset = aligned_offset;
    buffer_binding.SizeInBytes = byte_length;
    output_buffer_binding[name] = buffer_binding;
    output_binding_desc[i] = {DML_BINDING_TYPE_BUFFER,
                              &output_buffer_binding[name]};
    aligned_offset += Align(byte_length, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
    ++i;
  }

  execution_context_->ExecuteGraph(graph_id_, mCompiledOperator.Get(),
                                   input_binding_desc, output_binding_desc);

  auto named_outputs = ml::webnn::mojom::NamedOutputs::New();
  HRESULT hr = output_resource_readback_->ReadbackResource(named_outputs,
                                                           outputs_resource);
  if (FAILED(hr)) {
    std::move(callback).Run(ComputeResult::kUnknownError, nullptr);
    return;
  }

  std::move(callback).Run(ComputeResult::kOk, std::move(named_outputs));
}

}  // namespace content::webnn

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
#include "content/browser/ml/webnn/dml/utils_dml.h"
#include "content/browser/ml/webnn/fusion_operators.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

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
                                   const std::vector<UINT>& input_dims) {
  UINT nStride = 0, cStride = 0, hStride = 0, wStride = 0;
  switch (transposeType) {
    case NhwcToNchw:
      nStride = input_dims[1] * input_dims[2] * input_dims[3];
      hStride = input_dims[2] * input_dims[3];
      wStride = input_dims[3];
      cStride = 1;
      return {nStride, cStride, hStride, wStride};
    case NchwToNhwc:
      nStride = input_dims[1] * input_dims[2] * input_dims[3];
      cStride = input_dims[2] * input_dims[3];
      hStride = input_dims[3];
      wStride = 1;
      return {nStride, hStride, wStride, cStride};
    default:
      DCHECK(0);
      break;
  }
}

std::vector<UINT> transposeStridesToNchw(
    const std::vector<UINT>& input_dims,
    const DML_TENSOR_DESC* input_tensor_desc) {
  const DML_BUFFER_TENSOR_DESC* bufferDesc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(input_tensor_desc->Desc);
  DCHECK(bufferDesc != nullptr && bufferDesc->DimensionCount == 4);
  auto* strides = bufferDesc->Strides;
  if (strides != nullptr) {
    return {strides[0], strides[3], strides[1], strides[2]};
  } else {
    return transposeStrides(NhwcToNchw, input_dims);
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
      DCHECK(0);
  }
  dmlFusedOperatorDesc.Desc = &dmlActicationOperatorDesc;
  return &dmlFusedOperatorDesc;
}

std::vector<UINT> ExpandDimensions(const std::vector<UINT>& dims, size_t rank) {
  DCHECK(rank >= dims.size());
  std::vector<UINT> newDims(rank, 1);
  for (size_t i = 0; i < dims.size(); ++i) {
    newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
  }
  return newDims;
}

std::vector<UINT> transposeDimensions(TransposeType transposeType,
                                      const std::vector<UINT>& input_dims) {
  std::vector<UINT> newInputDims(4);
  switch (transposeType) {
    case NhwcToNchw:
      newInputDims[0] = input_dims[0];
      newInputDims[1] = input_dims[3];
      newInputDims[2] = input_dims[1];
      newInputDims[3] = input_dims[2];
      break;
    case NchwToNhwc:
      newInputDims[0] = input_dims[0];
      newInputDims[1] = input_dims[2];
      newInputDims[2] = input_dims[3];
      newInputDims[3] = input_dims[1];
      break;
    default:
      DCHECK(0);
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
      DCHECK(0);
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
      DCHECK(0);
      break;
  }
  return {oStride, iStride, hStride, wStride};
}

DML_TENSOR_DATA_TYPE GetTensorDataType(OperandType type) {
  DML_TENSOR_DATA_TYPE data_type;
  if (type == OperandType::kFloat32) {
    data_type = DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if (type == OperandType::kFloat16) {
    data_type = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if (type == OperandType::kInt32) {
    data_type = DML_TENSOR_DATA_TYPE_INT32;
  } else if (type == OperandType::kUint32) {
    data_type = DML_TENSOR_DATA_TYPE_UINT32;
  } else {
    LOG(ERROR) << "This data type is not supported";
    return DML_TENSOR_DATA_TYPE_UNKNOWN;
  }

  return data_type;
}

// Strides are used to express broadcasting (by specifying a stride of 0) as
// well as padding. If Strides is not specified, each dimension in the tensor is
// considered to be contiguously packed, with no additional padding. The
// calculated strides refer to
// https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides
std::vector<UINT> CalculateStridesForBroadcast(
    NodeOutput* node_output,
    std::vector<UINT> broadcasted_dims) {
  auto& tensor_desc = node_output->GetTensorDesc();
  auto original_dims = tensor_desc.GetDimensions();
  auto original_rank = original_dims.size(),
       broadcasted_rank = broadcasted_dims.size();
  std::vector<bool> broadcast_flags(broadcasted_rank, false);
  auto rank_gap = broadcasted_rank - original_rank;
  for (size_t i = 0; i < rank_gap; ++i) {
    broadcast_flags[i] = true;
  }
  for (size_t i = 0; i < original_rank; ++i) {
    if (original_dims[i] == 1 && broadcasted_dims[rank_gap + i] != 1) {
      broadcast_flags[rank_gap + i] = true;
    }
  }

  for (size_t i = 0; i < broadcasted_rank; ++i) {
    if (broadcast_flags[i]) {
      broadcasted_dims[i] = 1;
    }
  }
  std::vector<UINT> strides(broadcasted_rank);
  auto existed_strides = tensor_desc.GetStrides();
  if (existed_strides) {
    auto indexBegin = broadcasted_rank - original_rank;
    for (size_t i = 0, j = 0; i < broadcasted_rank; ++i) {
      if (i < indexBegin) {
        strides[i] = 0;
      } else {
        strides[i] = broadcast_flags[i] ? 0 : existed_strides.value()[j];
        ++j;
      }
    }
  } else {
    strides[broadcasted_rank - 1] =
        broadcast_flags[broadcasted_rank - 1] ? 0 : 1;
    size_t elements = 1;
    for (size_t i = 1; i < broadcasted_rank; i++) {
      size_t j = broadcasted_rank - i - 1;
      elements *= broadcasted_dims[j + 1];
      strides[j] = broadcast_flags[j] ? 0 : elements;
    }
  }
  return strides;
}

}  // namespace

#define DAWN_INTERNAL_ERROR(MESSAGE)            \
  do {                                          \
    error_messages_ = MESSAGE;                  \
    DCHECK(0);                                  \
    build_result_ = BuildResult::kUnknownError; \
    return;                                     \
  } while (0)

#define CREATE_BINARY_OPERATOR(type, a_tensor_desc, b_tensor_desc, \
                               output_tensor, node)                \
  DML_ELEMENT_WISE_##type##_OPERATOR_DESC operator_desc{};         \
  operator_desc.ATensor = a_tensor_desc;                           \
  operator_desc.BTensor = b_tensor_desc;                           \
  operator_desc.OutputTensor = output_tensor;                      \
  node = graph_desc_builder_->CreateOperatorNode(                  \
      DML_OPERATOR_ELEMENT_WISE_##type, &operator_desc);

#define CREATE_UNARY_OPERATOR(type, input_tensor_desc, node)          \
  DML_##type##_OPERATOR_DESC operator_desc{};                         \
  operator_desc.InputTensor = input_tensor_desc;                      \
  operator_desc.OutputTensor = input_tensor_desc;                     \
  node = graph_desc_builder_->CreateOperatorNode(DML_OPERATOR_##type, \
                                                 &operator_desc);

// Append IDENTITY to remove the strides of input tensor. Use this to implement
// Reshape, Squeeze, Transpose and avoid creating an invaild graph with input =
// output.
#define APPEND_IDENTITY(input_tensor, output_tensor, node) \
  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC operator_desc{}; \
  operator_desc.InputTensor = input_tensor;                \
  operator_desc.OutputTensor = output_tensor;              \
  node = graph_desc_builder_->CreateOperatorNode(          \
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
  Node input_node = graph_desc_builder_->CreateInputNode(std::move(name));
  TensorDesc tensor_desc(GetTensorDataType(desc->data_type),
                         ConvertDimensions(desc->dimensions));
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      input_node, 0, std::move(tensor_desc));
  node_output_map_[desc->object_id] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddConstant(OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  if (node_output_map_.find(desc->object_id) != node_output_map_.end()) {
    LOG(ERROR) << "There are issues in sorting graph";
    return;
  }
  Node constant_node = graph_desc_builder_->CreateConstantNode(desc->object_id);
  TensorDesc tensor_desc(GetTensorDataType(desc->data_type),
                         DML_TENSOR_FLAG_OWNED_BY_DML,
                         ConvertDimensions(desc->dimensions));
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      constant_node, 0, std::move(tensor_desc));
  node_output_map_[desc->object_id] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddElementWiseBinary(uint32_t a_id,
                                        uint32_t b_id,
                                        BinaryOperandType type,
                                        OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(a_id) != node_output_map_.end());
  DCHECK(node_output_map_.find(b_id) != node_output_map_.end());

  auto* a_node_output = node_output_map_[a_id].get();
  auto* b_node_output = node_output_map_[b_id].get();
  auto output_dims = ConvertDimensions(output_desc->dimensions);
  std::vector<UINT> output_new_dims = output_dims;

  auto a_broadcasted_strides =
      CalculateStridesForBroadcast(a_node_output, output_dims);
  auto& a_tensor_desc = a_node_output->GetTensorDesc();
  TensorDesc a_broadcasted_tensor(a_tensor_desc.GetDataType(),
                                  a_tensor_desc.GetFlags(), output_dims,
                                  a_broadcasted_strides);

  auto b_broadcasted_strides =
      CalculateStridesForBroadcast(b_node_output, output_dims);
  auto& b_tensor_desc = b_node_output->GetTensorDesc();
  TensorDesc b_broadcasted_tensor(b_tensor_desc.GetDataType(),
                                  b_tensor_desc.GetFlags(), output_dims,
                                  b_broadcasted_strides);

  TensorDesc output_tensor(b_tensor_desc.GetDataType(), output_new_dims);
  Node node;
  switch (type) {
    case BinaryOperandType::kAdd: {
      CREATE_BINARY_OPERATOR(ADD, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case BinaryOperandType::kDiv: {
      CREATE_BINARY_OPERATOR(DIVIDE, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case BinaryOperandType::kMul: {
      CREATE_BINARY_OPERATOR(MULTIPLY, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case BinaryOperandType::kSub: {
      CREATE_BINARY_OPERATOR(SUBTRACT, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case BinaryOperandType::kMax: {
      CREATE_BINARY_OPERATOR(MAX, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case BinaryOperandType::kMin: {
      CREATE_BINARY_OPERATOR(MIN, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    default:
      DAWN_INTERNAL_ERROR(" Binary op is not implemented.");
  }
  graph_desc_builder_->Connect({a_node_output, b_node_output}, {node});
  auto node_output =
      graph_desc_builder_->CreateNodeOutput(node, 0, std::move(output_tensor));
  node_output_map_[output_desc->object_id] = std::move(node_output);
  return;
}

std::unique_ptr<NodeOutput> GraphDMLImpl::Clamp(NodeOutput* input_node,
                                                const ClampOptions* options) {
  auto& input_tensor = input_node->GetTensorDesc();
  TensorDesc output_tensor(input_tensor.GetDataType(),
                           input_tensor.GetDimensions());
  DML_ELEMENT_WISE_CLIP_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor.Get();
  operator_desc.OutputTensor = output_tensor.Get();
  operator_desc.ScaleBias = nullptr;
  operator_desc.Min = options->minValue;
  operator_desc.Max = options->maxValue;
  Node operator_node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_ELEMENT_WISE_CLIP, &operator_desc);

  graph_desc_builder_->Connect({input_node}, {operator_node});
  return graph_desc_builder_->CreateNodeOutput(operator_node, 0,
                                               std::move(output_tensor));
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
                                        const std::vector<UINT>& input_dims) {
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
  auto& input_tensor_desc = input_node->GetTensorDesc();
  TensorDesc nhwc_tensor_desc(input_tensor_desc.GetDataType(),
                              input_tensor_desc.GetFlags(), nhwcOutputDims,
                              nhwcOutputStrides);
  auto node = input_node->GetNode();
  input_node = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(nhwc_tensor_desc));
  return;
}

void GraphDMLImpl::AddConv2d(uint32_t input_id,
                             uint32_t filter_id,
                             Conv2dOptionsPtr options,
                             OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(input_id) != node_output_map_.end());
  DCHECK(node_output_map_.find(filter_id) != node_output_map_.end());

  auto* input_node = node_output_map_[input_id].get();
  auto* filter_node = node_output_map_[filter_id].get();

  auto& input_node_desc = input_node->GetTensorDesc();
  auto input_dims = input_node_desc.GetDimensions();
  auto filterDims = filter_node->GetTensorDesc().GetDimensions();
  auto output_dims = ConvertDimensions(output_desc->dimensions);
  std::vector<UINT> input_nchw_dims = input_dims, filter_nchw_dims = filterDims,
                    output_nchw_dims = output_dims;

  DML_TENSOR_DESC* input_tensor_desc = input_node_desc.Get();
  TensorDesc nhwc_tensor_desc;
  if (options->inputLayout == InputOperandLayout::kNhwc) {
    input_nchw_dims = transposeDimensions(NhwcToNchw, input_dims);
    output_nchw_dims = transposeDimensions(NhwcToNchw, output_dims);
    auto input_nchw_Strides =
        transposeStridesToNchw(input_dims, input_tensor_desc);

    nhwc_tensor_desc =
        TensorDesc(input_node_desc.GetDataType(), input_node_desc.GetFlags(),
                   input_nchw_dims, input_nchw_Strides);
    input_tensor_desc = nhwc_tensor_desc.Get();
  }

  DML_TENSOR_DESC* filter_tensor_desc = filter_node->GetTensorDesc().Get();
  TensorDesc new_filter_tensor_desc;
  if (options->filterLayout != Conv2dFilterOperandLayout::kOihw) {
    filter_nchw_dims =
        transposeFilterDimensionsAsOihw(options->filterLayout, filterDims);
    auto filter_oihw_strides =
        transposeFilterStridesAsOihw(options->filterLayout, filterDims);

    auto& fileter_desc = filter_node->GetTensorDesc();
    new_filter_tensor_desc =
        TensorDesc(fileter_desc.GetDataType(), fileter_desc.GetFlags(),
                   filter_nchw_dims, filter_oihw_strides);
    filter_tensor_desc = new_filter_tensor_desc.Get();
  }

  std::vector<NodeOutput*> input_nodes = {input_node, filter_node};
  TensorDesc bias_tensor_desc;
  if (options->bias_id != 0) {
    DCHECK(node_output_map_.find(options->bias_id) != node_output_map_.end());
    auto* bias_node = node_output_map_[options->bias_id].get();
    auto& bias_desc = bias_node->GetTensorDesc();
    auto bias_dims = bias_desc.GetDimensions();
    if (bias_dims[0] != filter_nchw_dims[0] || bias_dims.size() != 1) {
      DAWN_INTERNAL_ERROR(
          "The bias should be 1-D tensor with the shape of [output_channels].");
    }

    // Reshape bias from 1-D to 4-D for NCHW layout.
    std::vector<UINT> bias_expand_dims = {1, bias_dims[0], 1, 1};
    bias_tensor_desc = TensorDesc(bias_desc.GetDataType(), bias_desc.GetFlags(),
                                  bias_expand_dims);
    input_nodes.push_back(bias_node);
  }

  // FIXME(nhu): strides, dilations, padding should be uint32_t
  // need to fix the spec.
  std::vector<UINT> strides = ConvertDimensions(options->strides);
  std::vector<UINT> dilations = ConvertDimensions(options->dilations);

  std::vector<UINT> padding =
      options->auto_pad == AutoPad::kExplicit
          ? ExplicitPadding<Conv2dOptions>(options.get())
          : ImplicitPadding<Conv2dOptions>(options.get(), input_nchw_dims,
                                           filter_nchw_dims);
  std::vector<UINT> startPadding = {padding[0], padding[2]};
  std::vector<UINT> endPadding = {padding[1], padding[3]};
  std::vector<UINT> defaultOutPadding = {0, 0};

  DML_ACTIVATION_LINEAR_OPERATOR_DESC dmlActicationOperatorDesc{};
  DML_OPERATOR_DESC dmlFusedOperatorDesc = {};
  DML_OPERATOR_DESC* fusedActivation =
      CreateFusedOperator(options->activation.get(), dmlActicationOperatorDesc,
                          dmlFusedOperatorDesc);

  TensorDesc output_tensor(input_node_desc.GetDataType(), output_nchw_dims);
  DML_CONVOLUTION_OPERATOR_DESC operator_desc{};
  operator_desc.InputTensor = input_tensor_desc;
  operator_desc.FilterTensor = filter_tensor_desc;
  operator_desc.BiasTensor = bias_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor.Get();

  operator_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
  operator_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
  operator_desc.DimensionCount = input_dims.size() - 2;
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
  auto output_node = graph_desc_builder_->CreateNodeOutput(
      operator_node, 0, std::move(output_tensor));

  // Transpose output from nchw->nhwc.
  if (options->inputLayout == InputOperandLayout::kNhwc) {
    TransposeOutputToNhwc(output_node, output_nchw_dims);
  }

  EmulateFusedOperator(options->activation.get(), output_node, output_dims);
  node_output_map_[output_desc->object_id] = std::move(output_node);
  return;
}

void GraphDMLImpl::AddReshape(uint32_t input_id,
                              OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(input_id) != node_output_map_.end());

  auto output_dims = ConvertDimensions(output_desc->dimensions);
  auto* input_node = node_output_map_[input_id].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  TensorDesc output_tensor(input_tensor_desc.GetDataType(),
                           input_tensor_desc.GetFlags(), output_dims);
  // Reshape is not a real node in DML, just need to update node output with new
  // tensor.
  auto node = input_node->GetNode();
  node_output_map_[output_desc->object_id] =
      graph_desc_builder_->CreateNodeOutput(node, 0, std::move(output_tensor));
  return;
}

void GraphDMLImpl::AddGemm(uint32_t a_id,
                           uint32_t b_id,
                           GemmOptionsPtr options,
                           OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(a_id) != node_output_map_.end());
  DCHECK(node_output_map_.find(b_id) != node_output_map_.end());

  // The shape of a tensor is 2D definited in WebNN Spec, but DML only support
  // 4D, so expand dimensions to 4D.
  // TODO: DML_FEATURE_LEVEL_4_0 and above support 2D.
  // DCHECK(a_dims.size() == 2);
  auto* a_node_output = node_output_map_[a_id].get();
  auto& a_tensor_desc = a_node_output->GetTensorDesc();
  auto a_expand_dims = ExpandDimensions(a_tensor_desc.GetDimensions(), 4);
  TensorDesc a_expand_tensor(a_tensor_desc.GetDataType(),
                             a_tensor_desc.GetFlags(), a_expand_dims);

  // DCHECK(b_dims.size() == 2);
  auto* b_node_output = node_output_map_[b_id].get();
  auto& b_tensor_desc = b_node_output->GetTensorDesc();
  auto b_expand_dims = ExpandDimensions(b_tensor_desc.GetDimensions(), 4);
  TensorDesc b_expand_tensor(b_tensor_desc.GetDataType(),
                             b_tensor_desc.GetFlags(), b_expand_dims);

  auto output_dims = ConvertDimensions(output_desc->dimensions);
  DCHECK(output_dims.size() == 2);
  auto output_expand_dims = ExpandDimensions(output_dims, 4);
  TensorDesc output_expand_tensor(b_tensor_desc.GetDataType(),
                                  output_expand_dims);

  // The operand c is optional.
  TensorDesc c_expand_tensor;
  std::vector<NodeOutput*> input_nodes = {a_node_output, b_node_output};
  if (options->c_id != 0) {
    DCHECK(node_output_map_.find(options->c_id) != node_output_map_.end());
    auto* c_node_output = node_output_map_[options->c_id].get();
    // It is either a scalar, or of the shape that is unidirectionally
    // broadcastable to the shape [M, N] definited in WebNN Spec, DML only
    // support 4D, so broadCast the Shape of optional C to {1, 1, M, N }
    // supported in DML.
    auto c_broadcasted_strides =
        CalculateStridesForBroadcast(c_node_output, output_expand_dims);
    auto& c_tensor_desc = c_node_output->GetTensorDesc();
    c_expand_tensor =
        TensorDesc(c_tensor_desc.GetDataType(), c_tensor_desc.GetFlags(),
                   output_expand_dims, c_broadcasted_strides);
    input_nodes.push_back(c_node_output);
  }

  DML_MATRIX_TRANSFORM aTranspose = options->a_transpose
                                        ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                        : DML_MATRIX_TRANSFORM_NONE;
  DML_MATRIX_TRANSFORM bTranspose = options->b_transpose
                                        ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                        : DML_MATRIX_TRANSFORM_NONE;
  DML_GEMM_OPERATOR_DESC gemm_desc = {};
  gemm_desc.ATensor = a_expand_tensor.Get();
  gemm_desc.BTensor = b_expand_tensor.Get();
  gemm_desc.CTensor = c_expand_tensor.Get();
  gemm_desc.OutputTensor = output_expand_tensor.Get();
  gemm_desc.TransA = aTranspose;
  gemm_desc.TransB = bTranspose;
  gemm_desc.Alpha = options->alpha;
  gemm_desc.Beta = options->beta;

  Node operator_node =
      graph_desc_builder_->CreateOperatorNode(DML_OPERATOR_GEMM, &gemm_desc);
  graph_desc_builder_->Connect(std::move(input_nodes), {operator_node});
  DCHECK_LT(output_dims.size(), output_expand_dims.size());
  TensorDesc output_tensor(b_tensor_desc.GetDataType(), output_dims);
  node_output_map_[output_desc->object_id] =
      graph_desc_builder_->CreateNodeOutput(operator_node, 0,
                                            std::move(output_tensor));
  return;
}

void GraphDMLImpl::AddPool2d(uint32_t input_id,
                             Pool2dOptionsPtr options,
                             Pool2dType type,
                             OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(input_id) != node_output_map_.end());

  auto* input_node = node_output_map_[input_id].get();
  auto& input_node_desc = input_node->GetTensorDesc();
  auto input_dims = input_node_desc.GetDimensions();
  auto output_dims = ConvertDimensions(output_desc->dimensions);
  std::vector<UINT> input_nchw_dims = input_dims,
                    output_nchw_dims = output_dims;

  DML_TENSOR_DESC* input_tensor_desc = input_node_desc.Get();
  TensorDesc nhwc_input_tensor;
  if (options->layout == InputOperandLayout::kNhwc) {
    input_nchw_dims = transposeDimensions(NhwcToNchw, input_dims);
    output_nchw_dims = transposeDimensions(NhwcToNchw, output_dims);
    auto input_nchw_strides =
        transposeStridesToNchw(input_dims, input_tensor_desc);

    nhwc_input_tensor =
        TensorDesc(input_node_desc.GetDataType(), input_node_desc.GetFlags(),
                   input_nchw_dims, input_nchw_strides);
    input_tensor_desc = nhwc_input_tensor.Get();
  }

  std::vector<UINT> strides = ConvertDimensions(options->strides);
  std::vector<UINT> dilations = ConvertDimensions(options->dilations);

  std::vector<UINT> windowSizes;
  if (!options->window_dimensions.empty()) {
    windowSizes = ConvertDimensions(options->window_dimensions);
  } else {
    windowSizes = {input_nchw_dims[2], input_nchw_dims[3]};
  }

  // TODO:: Support AutoPad::kSameUpper and kSameLower;
  auto padding = options->auto_pad == AutoPad::kExplicit
                     ? ExplicitPadding<Pool2dOptions>(options.get())
                     : ImplicitPadding<Pool2dOptions>(
                           options.get(), input_nchw_dims, windowSizes);
  std::vector<UINT> startPadding = {padding[0], padding[2]};
  std::vector<UINT> endPadding = {padding[1], padding[3]};

  TensorDesc output_tensor(input_node_desc.GetDataType(), output_nchw_dims);
  Node operator_node;
  if (type == Pool2dType::kAveragePool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      DAWN_INTERNAL_ERROR("The dilations of average pool2d are not supported.");
    }
    DML_AVERAGE_POOLING_OPERATOR_DESC dml_desc = {};
    dml_desc.InputTensor = input_tensor_desc;
    dml_desc.OutputTensor = output_tensor.Get();
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
    dml_desc.InputTensor = input_tensor_desc;
    dml_desc.OutputTensor = output_tensor.Get();
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
            output_nchw_dims[2 + i] + startPadding[i] + endPadding[i];
        uint32_t dilatedWindowSize = 1 + (windowSizes[i] - 1) * dilations[i];
        output_nchw_dims[2 + i] =
            (dilatedWindowSize >= paddedInputSize)
                ? 1
                : (paddedInputSize - dilatedWindowSize) / strides[i] + 1;
      }
    }

    output_tensor = TensorDesc(input_node_desc.GetDataType(), output_nchw_dims);
    DML_MAX_POOLING2_OPERATOR_DESC desc = {};
    desc.InputTensor = input_tensor_desc;
    desc.OutputTensor = output_tensor.Get();
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
  auto output_node = graph_desc_builder_->CreateNodeOutput(
      operator_node, 0, std::move(output_tensor));

  // Transpose output from nchw->nhwc.
  if (options->layout == InputOperandLayout::kNhwc) {
    TransposeOutputToNhwc(output_node, output_nchw_dims);
  }

  node_output_map_[output_desc->object_id] = std::move(output_node);

  return;
}

void GraphDMLImpl::AddUnary(uint32_t input_id,
                            UnaryOperandType type,
                            OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(input_id) != node_output_map_.end());

  auto* input_node = node_output_map_[input_id].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  Node node;
  switch (type) {
    case UnaryOperandType::kRelu: {
      CREATE_UNARY_OPERATOR(ACTIVATION_RELU, input_tensor_desc.Get(), node);
    } break;
    case UnaryOperandType::kSigmoid: {
      CREATE_UNARY_OPERATOR(ACTIVATION_SIGMOID, input_tensor_desc.Get(), node);
    } break;
    case UnaryOperandType::kSoftmax: {
      CREATE_UNARY_OPERATOR(ACTIVATION_SOFTMAX, input_tensor_desc.Get(), node);
    } break;
    default:
      DCHECK(0);
      break;
  }
  graph_desc_builder_->Connect({input_node}, {node});
  TensorDesc output_tensor_desc(input_tensor_desc.GetDataType(),
                                input_tensor_desc.GetDimensions());
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_desc->object_id] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddFusionClamp(ClampOptionsPtr options,
                                  uint32_t operator_id) {
  fusion_operators_->AddClampOption(operator_id, std::move(options));
}

void GraphDMLImpl::AddOutput(const std::string& name, uint32_t operand_id) {
  DCHECK(node_output_map_.find(operand_id) != node_output_map_.end());
  auto* output_node = node_output_map_[operand_id].get();
  DCHECK(output_node != nullptr);

  // Append identity to avoid directly using graph input as output, and
  // avoid lack of considering the impacts of strides if there are.
  auto node = output_node->GetNode();
  if (node.type == NodeType::kInput || node.type == NodeType::kConstant ||
      output_node->GetTensorDesc().GetStrides()) {
    auto& input_tensor = output_node->GetTensorDesc();

    TensorDesc output_tensor(input_tensor.GetDataType(),
                             input_tensor.GetDimensions());
    APPEND_IDENTITY(input_tensor.Get(), output_tensor.Get(), node);
    graph_desc_builder_->Connect({output_node}, {node});
    std::unique_ptr<NodeOutput> identity_output_node =
        graph_desc_builder_->CreateNodeOutput(node, 0,
                                              std::move(output_tensor));
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
  ComPtr<gpgmm::d3d12::ResourceAllocation> constants_resource = nullptr;
  if (constants_info.get() != nullptr) {
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        constants_info->shared_memory;
    size_t constants_byte_length = shared_memory_region.GetSize();
    ExecutionResources* execution_resources =
        execution_context_->GetExecutionResources();
    constants_resource = execution_resources->Allocate(constants_byte_length);
    uploader->UploadConstants(constants_resource->GetResource(),
                              constants_info);
  }

  auto input_nodes = graph_desc_builder_->GetInputNodes();
  std::vector<DML_BUFFER_BINDING> input_buffer_binding(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto input = input_nodes[i];
    if (input.type == NodeType::kConstant) {
      input_buffer_binding[i].Buffer = constants_resource->GetResource();
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

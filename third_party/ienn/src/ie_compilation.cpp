// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ie_compilation.h"

#include <string>
#include <utility>

#include "constants.h"
#include "transpose_sinking.h"
#include "utils.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"
namespace InferenceEngine {
using namespace ngraph;
namespace {

std::vector<size_t> ConvertDims(const std::vector<uint32_t>& dimensions) {
  // IE default layout is NCHW
  std::vector<size_t> dims(dimensions.size(), 0);
  if (dimensions.size() == 1) {
    dims[0] = dimensions[0];
  } else if (dimensions.size() == 2) {
    dims[0] = dimensions[0];
    dims[1] = dimensions[1];
  } else if (dimensions.size() == 3) {
    // HWC -> CHW
    dims[0] = dimensions[2];
    dims[1] = dimensions[0];
    dims[2] = dimensions[1];
  } else if (dimensions.size() == 4) {
    // NHWC -> NCHW
    dims[0] = dimensions[0];
    dims[1] = dimensions[3];
    dims[2] = dimensions[1];
    dims[3] = dimensions[2];
  }
  return dims;
}

template <typename T1, typename T2>
int32_t CopyDataToBuffer(T1* dst,
                         const T2* src,
                         const std::vector<uint32_t>& dims,
                         bool reorder) {
  uint32_t result;
  if (reorder) {
    result = Reorder<T1, T2>(dst, src, dims);
  } else {
    result = CloneData<T1, T2>(dst, src, dims);
  }
  return result;
}

Shape CreateShape(const std::vector<uint32_t>& dimensions) {
  std::vector<size_t> dims;
  dims.reserve(dimensions.size());
  for (auto dim : dimensions) {
    dims.push_back(dim);
  }
  return Shape(dims);
}

// Transpose NHWC <=> NCHW.
ngraph::Output<ngraph::Node> TransposeLayout(ngraph::Output<ngraph::Node> node,
                                             bool nhwc_to_nchw,
                                             bool nghwc_to_ngchw = false) {
  AxisVector order = nghwc_to_ngchw ? AxisVector{0, 1, 4, 2, 3}
                                    : nhwc_to_nchw ? AxisVector{0, 3, 1, 2}
                                                   : AxisVector{0, 2, 3, 1};
  const auto order_node =
      op::Constant::create(element::i64, Shape{order.size()}, order);
  auto transpose_node = std::make_shared<op::v1::Transpose>(node, order_node);
  return transpose_node->output(0);
}

std::shared_ptr<ngraph::Node> AddActivationByFusedCode(
    int32_t fuse_code,
    std::shared_ptr<ngraph::Node> input_node) {
  std::shared_ptr<ngraph::Node> activation_node;
  if (fuse_code == fuse_t::FUSED_RELU) {
    activation_node = std::make_shared<op::v0::Relu>(input_node->output(0));
  } else if (fuse_code == fuse_t::FUSED_RELU1) {
    activation_node =
        std::make_shared<op::v0::Clamp>(input_node->output(0), -1.0, 1.0);
  } else if (fuse_code == fuse_t::FUSED_RELU6) {
    activation_node =
        std::make_shared<op::v0::Clamp>(input_node->output(0), 0.0, 6.0);
  } else if (fuse_code == fuse_t::FUSED_NONE) {
    return input_node;
  }
  return activation_node;
}

ngraph::Output<ngraph::Node> Reshape(ngraph::Output<ngraph::Node>& input_node,
                                     std::vector<size_t> new_shape) {
  auto target_shape_node = std::make_shared<op::Constant>(
      element::i64, Shape{new_shape.size()}, new_shape);
  auto reshape_node = std::make_shared<op::v1::Reshape>(
      input_node, target_shape_node->output(0), true);
  return reshape_node->output(0);
}

}  // namespace

Compilation::Compilation(ModelInfoPtr model)
    : model_(model), network_(nullptr) {}

void Compilation::SetPreference(int32_t prefer) {
  preference_ = prefer;
}

int32_t Compilation::GetPreference() {
  return preference_;
}

int32_t Compilation::Compile() {
  int32_t result;
  for (size_t i = 0; i < model_->inputs.size(); ++i) {
    result = AddInput(model_->inputs[i]);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  for (size_t i = 0; i < model_->operations.size(); ++i) {
    const Operation& operation = model_->operations[i];
    const int32_t type = operation.type;

    if (operation.outputs.size() != 1) {
      std::cout << "Only 1 output is supported";
      return error_t::BAD_DATA;
    }

    int32_t result = error_t::NOT_ERROR;
    if (type == operation_t::ADD || type == operation_t::MUL) {
      result = AddElementwise(operation);
    } else if (type == operation_t::CONV_2D ||
               type == operation_t::DEPTHWISE_CONV_2D ||
               type == operation_t::ATROUS_CONV_2D ||
               type == operation_t::ATROUS_DEPTHWISE_CONV_2D) {
      result = AddConvolution(operation);
    } else if (type == operation_t::AVERAGE_POOL_2D ||
               type == operation_t::MAX_POOL_2D) {
      result = AddPooling(operation);
    } else if (type == operation_t::SOFTMAX) {
      result = AddSoftmax(operation);
    } else if (type == operation_t::CONVERT) {
      result = AddConvert(operation);
    } else if (type == operation_t::RESHAPE) {
      result = AddReshape(operation);
    } else if (type == operation_t::CONCATENATION) {
      result = AddConcatenation(operation);
    } else if (type == operation_t::FULLY_CONNECTED) {
      result = AddFullyConnected(operation);
    } else if (type == operation_t::RESIZE_BILINEAR_NN) {
      result = AddResizeBilinear(operation);
    } else if (type == operation_t::LOGISTIC) {
      result = AddSigmoid(operation);
    } else if (type == operation_t::ARGMAX) {
      result = AddArgmax(operation);
    } else if (type == operation_t::RELU) {
      result = AddActivation(operation);
    } else if (type == operation_t::TRANSPOSE) {
      result = AddTranspose(operation);
    } else if (type == operation_t::FAKE_QUANTIZE) {
      result = AddFakequantize(operation);
    } else {
      std::cout << "Operation type " << type << " is not supported."
                << std::endl;
      return error_t::BAD_DATA;
    }

    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  for (size_t i = 0; i < model_->outputs.size(); ++i) {
    result = AddOutput(model_->outputs[i]);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  try {
    std::shared_ptr<ngraph::Function> ngraph_function;
    ngraph_function =
        std::make_shared<Function>(ngraph_outputs_, ngraph_inputs_);
    // If the model only has one op, transpose_sinking will not needed.
    if (model_->operations.size() > 1) {
      ngraph::pass::Manager passes;
      passes.register_pass<tensorflow::ngraph_bridge::pass::TransposeSinking>();
      passes.run_passes(ngraph_function);
    }

    network_.reset(new CNNNetwork(ngraph_function));
    InputsDataMap input_info(network_->getInputsInfo());
    for (auto itr : input_info) {
      itr.second->setPrecision(Precision::FP32);
    }
  } catch (const std::exception& ex) {
    std::cout << "[IE] exception " << ex.what();
    return error_t::OP_FAILED;
  }

  return error_t::NOT_ERROR;
}

int32_t Compilation::AddInput(uint32_t index) {
  try {
    // MYRIAD only supports FP16 models, but it can accept FP32 inputs. So we
    // can also set the inputs precison to FP32.
    const Operand& operand = model_->operands[index];
    auto input_node = std::make_shared<op::Parameter>(
        preference_ != PREFER_LOW_POWER ? element::f32 : element::f16,
        CreateShape(operand.dimensions));
    index_op_map_[index] = input_node->output(0);
    ngraph_inputs_.push_back(input_node);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add input layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddOutput(uint32_t index) {
  if (index_op_map_.find(index) == index_op_map_.end()) {
    std::cout << "Layer for output index " << index << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    auto output_node = std::make_shared<op::Result>(index_op_map_[index]);
    ngraph_outputs_.push_back(output_node);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add output layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConstant(uint32_t index,
                                 bool reorder,
                                 std::vector<size_t> specified_dims) {
  try {
    if (index_op_map_.find(index) != index_op_map_.end()) {
      std::cout << "constant has been already created" << std::endl;
      return error_t::NOT_ERROR;
    }
    const Operand& operand = model_->operands[index];
    SizeVector dims = reorder ? ConvertDims(operand.dimensions)
                              : CreateShape(operand.dimensions);
    bool fp16_precision = preference_ == PREFER_LOW_POWER;
    Blob::Ptr blob;

    if (fp16_precision) {
      // MYRIAD only accepts FP16 precision.
      blob = make_shared_blob<int16_t>({Precision::FP16, dims, Layout::ANY});
    } else if (operand.type == data_t::TENSOR_FLOAT32) {
      // GNA only accepts FP32 precision, cpu/gpu use FP32 currently.
      blob = make_shared_blob<float>({Precision::FP32, dims, Layout::ANY});
    } else if (operand.type == data_t::TENSOR_QUANT8_ASYMM) {
      // cpu quant model accepts uint8 weights
      blob = make_shared_blob<uint8_t>({Precision::U8, dims, Layout::ANY});
    }
    blob->allocate();

    std::shared_ptr<op::Constant> constant_node;
    uint32_t result;
    SizeVector const_dims = specified_dims.size() == 0 ? dims : specified_dims;
    if (fp16_precision) {
      const float* src =
          reinterpret_cast<const float*>(model_->values[index].buffer);
      int16_t* dst = blob->buffer().as<int16_t*>();
      result = CopyDataToBuffer<int16_t, float>(dst, src, operand.dimensions,
                                                reorder);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
      constant_node = std::make_shared<op::Constant>(
          element::f16, Shape(const_dims), blob->buffer().as<int16_t*>());
    } else if (operand.type == data_t::TENSOR_FLOAT32) {
      const float* src =
          reinterpret_cast<const float*>(model_->values[index].buffer);
      float* dst = blob->buffer().as<float*>();
      result =
          CopyDataToBuffer<float, float>(dst, src, operand.dimensions, reorder);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
      // The layout of constant_node depends on the shape we set
      // e.g. 4D shape will use default NCHW
      constant_node = std::make_shared<op::Constant>(
          element::f32, Shape(const_dims), blob->buffer().as<float*>());
    } else if (operand.type == data_t::TENSOR_QUANT8_ASYMM) {
      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(model_->values[index].buffer);
      uint8_t* dst = blob->buffer().as<uint8_t*>();
      result = CopyDataToBuffer<uint8_t, uint8_t>(dst, src, operand.dimensions,
                                                  reorder);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
      constant_node = std::make_shared<op::Constant>(
          element::u8, Shape(const_dims), blob->buffer().as<float*>());
    }
    index_op_map_[index] = constant_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add constant " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddElementwise(const Operation& operation) {
  // Setup element-wise parameters.
  ElementWiseParams params;
  int32_t result = GetElementWiseParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  for (size_t i = 0; i < 2; ++i) {
    const uint32_t input_index = operation.inputs[i];
    if (index_op_map_.find(input_index) == index_op_map_.end()) {
      // Setup constants
      if (model_->values.find(input_index) != model_->values.end()) {
        result = AddConstant(input_index, false);
        if (result != error_t::NOT_ERROR) {
          return result;
        }
      } else {
        std::cout << "The layer for operand index " << input_index
                  << " is not ready";
        return error_t::BAD_DATA;
      }
    }
  }
  try {
    std::shared_ptr<ngraph::Node> elementwise_node;
    auto input0 = index_op_map_[operation.inputs[0]];
    auto input1 = index_op_map_[operation.inputs[1]];
    if (operation.type == ADD) {
      elementwise_node = std::make_shared<op::v1::Add>(input0, input1);
    } else if (operation.type == MUL) {
      elementwise_node = std::make_shared<op::v1::Multiply>(input0, input1);
    } else {
      std::cout << "Operation type " << operation.type << " is not supported";
      return error_t::BAD_DATA;
    }
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] =
        AddActivationByFusedCode(params.fuse_code, elementwise_node)->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add elementwise layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConvolution(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  const Operand& input = model_->operands[input_index];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  // Setup convolution params.
  ConvParams params;
  int32_t result = GetConvParamsV1(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  try {
    std::vector<ptrdiff_t> padBegin{params.padding_top, params.padding_left};
    std::vector<ptrdiff_t> padEnd{params.padding_bottom, params.padding_right};
    std::vector<size_t> strides{params.stride_height, params.stride_width};
    std::vector<size_t> dilations{params.dilation_height,
                                  params.dilation_width};
    const uint32_t weights_index = operation.inputs[1];
    const uint32_t bias_index = operation.inputs[2];
    std::shared_ptr<ngraph::Node> conv_node;

    auto conv_input_node =
        params.nhwc_layout ? TransposeLayout(index_op_map_[input_index], true)
                           : index_op_map_[input_index];
    if (params.depthwise) {
      if (index_op_map_.find(weights_index) == index_op_map_.end()) {
        // Setup constants
        SizeVector weights_dims{params.depth_out, 1, 1, params.filter_height,
                                params.filter_width};
        result = AddConstant(weights_index, params.nhwc_layout, weights_dims);
        if (result != error_t::NOT_ERROR) {
          return result;
        }
      } else {
        if (params.nhwc_layout) {
          // transpose the weights from nghwc to ngchw
          index_op_map_[weights_index] =
              TransposeLayout(index_op_map_[weights_index], false, true);
        }
      }
      conv_node = std::make_shared<op::v1::GroupConvolution>(
          conv_input_node, index_op_map_[weights_index], Strides(strides),
          CoordinateDiff(padBegin), CoordinateDiff(padEnd), Strides(dilations));
    } else {
      if (index_op_map_.find(weights_index) == index_op_map_.end()) {
        // Setup constants
        result = AddConstant(weights_index, params.nhwc_layout);
        if (result != error_t::NOT_ERROR) {
          return result;
        }
      } else {
        if (params.nhwc_layout) {
          // transpose the weights from nhwc to nchw
          index_op_map_[weights_index] =
              TransposeLayout(index_op_map_[weights_index], true);
        }
      }
      conv_node = std::make_shared<op::v1::Convolution>(
          conv_input_node, index_op_map_[weights_index], Strides(strides),
          CoordinateDiff(padBegin), CoordinateDiff(padEnd), Strides(dilations));
    }
    // add bias and fused-code
    const uint32_t output_index = operation.outputs[0];
    // Since cldnn doesn't support mixed layout
    // We reshape the bias to 4D NCHW aligned with conv_node
    // Setup constants
    size_t bias_rank = model_->operands[bias_index].dimensions.size();
    result =
        AddConstant(bias_index, bias_rank == 1 ? false : params.nhwc_layout,
                    bias_rank == 1 ? SizeVector({1, params.bias_length, 1, 1})
                                   : SizeVector({}));
    if (result != error_t::NOT_ERROR) {
      return result;
    }

    auto add_node = std::make_shared<op::v1::Add>(conv_node->output(0),
                                                  index_op_map_[bias_index]);
    auto fused_node =
        AddActivationByFusedCode(params.fuse_code, add_node)->output(0);
    index_op_map_[output_index] =
        params.nhwc_layout ? TransposeLayout(fused_node, false) : fused_node;
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add convolution layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddPooling(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  // Setup pooling params.
  PoolingParams params;
  int32_t result = GetPoolingParamsV1(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  try {
    auto pooling_input_node =
        params.nhwc_layout ? TransposeLayout(index_op_map_[input_index], true)
                           : index_op_map_[input_index];
    const uint32_t output_index = operation.outputs[0];
    std::shared_ptr<ngraph::Node> pooling_node;

    if (operation.type == operation_t::MAX_POOL_2D) {
      pooling_node = std::make_shared<op::v1::MaxPool>(
          pooling_input_node,
          std::vector<size_t>{params.stride_width, params.stride_height},
          Shape{params.padding_top, params.padding_left},
          Shape{params.padding_bottom, params.padding_right},
          std::vector<size_t>{params.filter_height, params.filter_width},
          op::RoundingType::FLOOR, op::PadType::EXPLICIT);
    } else {
      pooling_node = std::make_shared<op::v1::AvgPool>(
          pooling_input_node,
          std::vector<size_t>{params.stride_width, params.stride_height},
          Shape{params.padding_top, params.padding_left},
          Shape{params.padding_bottom, params.padding_right},
          std::vector<size_t>{params.filter_height, params.filter_width}, true,
          op::RoundingType::FLOOR, op::PadType::EXPLICIT);
    }
    auto fused_node =
        AddActivationByFusedCode(params.fuse_code, pooling_node)->output(0);
    index_op_map_[output_index] =
        params.nhwc_layout ? TransposeLayout(fused_node, false) : fused_node;
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add pooling layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConvert(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    // Setup constants
    if (model_->values.find(input_index) != model_->values.end()) {
      int32_t result = AddConstant(input_index, false);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
    } else {
      std::cout << "The layer for operand index " << input_index
                << " is not ready";
      return error_t::BAD_DATA;
    }
  }

  const uint32_t output_index = operation.outputs[0];
  int32_t output_type = model_->operands[output_index].type;
  ngraph::element::Type destination_type;
  switch (output_type) {
    case data_t::FLOAT32:
      destination_type = element::f32;
      break;
    case data_t::INT32:
      destination_type = element::i32;
      break;
    case data_t::TENSOR_QUANT8_ASYMM_SIGNED:
      destination_type = element::i8;
      break;
    case data_t::TENSOR_QUANT8_ASYMM:
      destination_type = element::u8;
      break;
    default:
      destination_type = element::f32;
  }

  try {
    auto convert_node = std::make_shared<op::v0::Convert>(
        index_op_map_[input_index], destination_type);
    index_op_map_[output_index] = convert_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add Convert node " << ex.what() << std::endl;
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddSoftmax(const Operation& operation) {
  // Setup softmax params.
  SoftmaxParams params;
  int32_t result = GetSoftmaxParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  // Check beta.
  if (params.beta != 1.0) {
    std::cout << "beta " << params.beta << " is not supported.";
    return error_t::BAD_DATA;
  }
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }

  const uint32_t output_index = operation.outputs[0];
  // For Android NNAPI, index = 2: An optional ANEURALNETWORKS_INT32 scalar,
  // default to -1, specifying the dimension the activation would be performed
  // on. Negative index is used to specify axis from the end (e.g. -1 for the
  // last axis).
  // For IE, There is no scaling factor, It's axis position (0-based) on which
  // to calculate the softmax.
  // For new Spec, it only support 2-D input tensor along axis 1.
  auto shape = index_op_map_[input_index].get_shape();
  // Only Android NNAPI support 4 rank to pass android test case.
  size_t axis;
  if (operation.inputs.size() > 2) {
    axis = GetScalarInt32(model_, operation.inputs[2]);
  } else {
    axis = shape.size() == 4 ? 3 : 1;
  }
  try {
    auto softmax_node =
        std::make_shared<op::v1::Softmax>(index_op_map_[input_index], axis);
    index_op_map_[output_index] = softmax_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add softmax node " << ex.what() << std::endl;
    return error_t::OP_FAILED;
  }

  return error_t::NOT_ERROR;
}

int32_t Compilation::AddReshape(const Operation& operation) {
  ReshapeParams params;
  int32_t result = GetReshapeParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;

  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] =
        Reshape(index_op_map_[input_index], params.new_shape);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add reshape layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConcatenation(const Operation& operation) {
  // Setup concatenation params.
  ConcatParams params;
  int32_t result = GetConcatParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;

  std::vector<std::shared_ptr<ngraph::Node>> input_nodes;
  for (size_t i = 0; i < operation.inputs.size() - 1; ++i) {
    const uint32_t input_index = operation.inputs[i];
    if (index_op_map_.find(input_index) == index_op_map_.end()) {
      // Setup constants
      if (model_->values.find(input_index) != model_->values.end()) {
        result = AddConstant(input_index, false);
        if (result != error_t::NOT_ERROR) {
          return result;
        }
      } else {
        std::cout << "The layer for operand index " << input_index
                  << " is not ready" << std::endl;
        return error_t::BAD_DATA;
      }
    }
    input_nodes.push_back(index_op_map_[input_index].get_node_shared_ptr());
  }
  const uint32_t output_index = operation.outputs[0];
  try {
    auto concat_node =
        std::make_shared<ngraph::op::Concat>(input_nodes, params.axis);
    index_op_map_[output_index] = concat_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add concat layer " << ex.what() << std::endl;
    return error_t::OP_FAILED;
  }

  return error_t::NOT_ERROR;
}

int32_t Compilation::AddFullyConnected(const Operation& operation) {
  // The rank of inputs is 4 for FullyConnected Android NN API, we extend a rank
  // to identify V2 Spec to implement MatMul Operations.
  if (operation.inputs.size() == 4) {
    return AddFullyConnectedV1(operation);
  }
  const uint32_t primary_index = operation.inputs[0];
  if (index_op_map_.find(primary_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << primary_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    auto primary_node = index_op_map_[primary_index];
    auto primary_shape = primary_node.get_shape();
    if (primary_shape.size() == 1) {
      // If a is 1-D, it is converted to a 2-D tensor by prepending a 1 to its
      // dimensions.
      primary_node = Reshape(primary_node, {1, primary_shape[0]});
    }

    const uint32_t secondary_index = operation.inputs[1];
    if (model_->values.find(secondary_index) != model_->values.end()) {
      AddConstant(secondary_index, false);
    }
    auto secondary_node = index_op_map_[secondary_index];
    auto secondary_shape = secondary_node.get_shape();
    if (secondary_shape.size() == 1) {
      // If b is 1-D, it is converted to a 2-D tensor by by appending a 1 to its
      // dimensions.
      secondary_node = Reshape(secondary_node, {secondary_shape[0], 1});
    }
    auto matmul_node = std::make_shared<op::v0::MatMul>(
        primary_node, secondary_node, false, false);
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] = matmul_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add fc layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddFullyConnectedV1(const Operation& operation) {
  // Setup fully connected params.
  FullyConnectedParams params;
  int32_t result = GetFullyConnectedParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t weights_index = operation.inputs[1];
    const uint32_t bias_index = operation.inputs[2];
    // Setup constants
    AddConstant(weights_index, false);
    AddConstant(bias_index, false);

    std::vector<size_t> dims{params.input_batch_size, params.input_size};
    // Need to transpose the weights.
    auto matmul_node = std::make_shared<op::v0::MatMul>(
        Reshape(index_op_map_[input_index], dims), index_op_map_[weights_index],
        false, true);
    auto add_node = std::make_shared<op::v1::Add>(matmul_node->output(0),
                                                  index_op_map_[bias_index]);
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] =
        AddActivationByFusedCode(params.fuse_code, add_node)->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add fc layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddResizeBilinear(const Operation& operation) {
  // Setup resize bilinear params.
  ResizeBilinearParams params;
  int32_t result = GetResizeBilinearParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;

  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready" << std::endl;
    return error_t::BAD_DATA;
  }

  ngraph::op::InterpolateAttrs attrs;
  attrs.axes = {2, 3};
  attrs.mode = "linear";
  attrs.align_corners = params.align_corners;
  attrs.antialias = false;
  attrs.pads_begin = {0, 0, 0, 0};
  attrs.pads_end = {0, 0, 0, 0};

  auto target_shape_node = op::Constant::create<int64_t>(
      element::i64, Shape{2}, {params.new_height, params.new_width});
  auto resize_input_node =
      params.nhwc_layout ? TransposeLayout(index_op_map_[input_index], true)
                         : index_op_map_[input_index];
  try {
    const uint32_t output_index = operation.outputs[0];
    auto resize_node =
        std::make_shared<op::v0::Interpolate>(
            resize_input_node, target_shape_node->output(0), attrs)
            ->output(0);
    index_op_map_[output_index] =
        params.nhwc_layout ? TransposeLayout(resize_node, false) : resize_node;
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add resize bilinear layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddArgmax(const Operation& operation) {
  ArgmaxParams params;
  int32_t result = GetArgmaxParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return error_t::BAD_DATA;

  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready" << std::endl;
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation.outputs[0];
    const auto k = op::Constant::create(element::i64, Shape{}, {1});
    auto topk_node = std::make_shared<op::v1::TopK>(
        index_op_map_[input_index], k->output(0), params.axis, "max", "value");
    // output(1) with top k indices for each slice along axis dimension
    index_op_map_[output_index] = topk_node->output(1);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add argmax layer " << ex.what() << std::endl;
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddSigmoid(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    auto sigmoid_node =
        std::make_shared<op::v0::Sigmoid>(index_op_map_[input_index]);
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] = sigmoid_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add sigmoid layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddActivation(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  auto relu_node = std::make_shared<op::v0::Relu>(index_op_map_[input_index]);
  const uint32_t output_index = operation.outputs[0];
  index_op_map_[output_index] = relu_node->output(0);
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddTranspose(const Operation& operation) {
  int32_t result;
  const uint32_t input_index = operation.inputs[0];
  if (model_->values.find(input_index) != model_->values.end()) {
    result = AddConstant(input_index, false);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  TransposeParams params;
  auto input_node = index_op_map_[input_index];
  result = GetTransposeParams(model_, operation, input_node, params);
  if (result != error_t::NOT_ERROR)
    return error_t::BAD_DATA;

  const auto order_node = op::Constant::create(
      element::i64, Shape{params.permutation.size()}, params.permutation);
  auto transpose_node =
      std::make_shared<op::v1::Transpose>(input_node, order_node);
  const uint32_t output_index = operation.outputs[0];
  index_op_map_[output_index] = transpose_node->output(0);
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddFakequantize(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t input_low_index = operation.inputs[1];
    const uint32_t input_high_index = operation.inputs[2];
    const uint32_t output_low_index = operation.inputs[3];
    const uint32_t output_high_index = operation.inputs[4];

    AddConstant(input_low_index, false);
    AddConstant(input_high_index, false);
    AddConstant(output_low_index, false);
    AddConstant(output_high_index, false);

    size_t levels = GetScalarInt32(model_, operation.inputs[5]);

    auto fakequantize_node = std::make_shared<op::v0::FakeQuantize>(
        index_op_map_[input_index], index_op_map_[input_low_index],
        index_op_map_[input_high_index], index_op_map_[output_low_index],
        index_op_map_[output_high_index], levels);

    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] = fakequantize_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add fakequantize layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

}  // namespace InferenceEngine

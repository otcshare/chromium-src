// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/graph_desc_builder.h"

namespace content::webnn {

GraphDescBuilder::GraphDescBuilder(ComPtr<IDMLDevice> device)
    : device_(device) {}

GraphDescBuilder::~GraphDescBuilder() = default;

Node GraphDescBuilder::CreateInputNode(std::string name) {
  InputNode node = {};
  node.name = std::move(name);
  node.type = NodeType::kInput;
  // The input index is increased as input node is added.
  uint32_t input_index = input_nodes_.size();
  node.index = input_index;
  input_nodes_.push_back(node);

  return node;
}

Node GraphDescBuilder::CreateConstantNode(uint32_t object_id) {
  InputNode node = {};
  node.object_id = object_id;
  node.type = NodeType::kConstant;
  // The input index is increased as input node is added.
  uint32_t input_index = input_nodes_.size();
  node.index = input_index;
  input_nodes_.push_back(node);

  return node;
}

Node GraphDescBuilder::CreateOperatorNode(DML_OPERATOR_TYPE type,
                                          const void* operator_desc) {
  DML_OPERATOR_DESC op_desc = {type, operator_desc};
  Microsoft::WRL::ComPtr<IDMLOperator> op;
  HRESULT hr = device_->CreateOperator(&op_desc, IID_PPV_ARGS(&op));
  if (FAILED(hr)) {
    return {NodeType::kUnknow, 0};
  }

  OperatorNode op_node = {};
  op_node.op = std::move(op);
  op_node.type = NodeType::kOperator;
  // The node index is increased as operator node is added.
  uint32_t node_index = operator_nodes_.size();
  op_node.index = node_index;
  operator_nodes_.push_back(std::move(op_node));
  DML_OPERATOR_GRAPH_NODE_DESC operator_node_desc = {};
  operator_node_desc.Operator = op_node.op.Get();
  graph_desc_.nodes.push_back(operator_node_desc);
  return op_node;
}

std::unique_ptr<NodeOutput> GraphDescBuilder::CreateNodeOutput(
    Node node,
    uint32_t output_index,
    TensorDesc tensor_desc) {
  return std::make_unique<NodeOutput>(node, output_index,
                                      std::move(tensor_desc));
}

void GraphDescBuilder::Connect(std::vector<NodeOutput*> inputs,
                               Node operator_node) {
  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    auto* node_output = inputs[input_index];
    auto input_node = node_output->GetNode();
    if (input_node.type == NodeType::kInput ||
        input_node.type == NodeType::kConstant) {
      DML_INPUT_GRAPH_EDGE_DESC input_edge = {};
      input_edge.GraphInputIndex = input_node.index;
      input_edge.ToNodeIndex = operator_node.index;
      input_edge.ToNodeInputIndex = input_index;

      graph_desc_.input_edges.push_back(input_edge);
    } else if (input_node.type == NodeType::kOperator) {
      DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediate_edge = {};
      intermediate_edge.FromNodeIndex = input_node.index;
      intermediate_edge.FromNodeOutputIndex = node_output->GetOutputIndex();
      intermediate_edge.ToNodeIndex = operator_node.index;
      intermediate_edge.ToNodeInputIndex = input_index;

      graph_desc_.intermediate_edges.push_back(intermediate_edge);
    }
  }
}

void GraphDescBuilder::AddOutputEdge(NodeOutput* node_output,
                                     std::string name) {
  DML_OUTPUT_GRAPH_EDGE_DESC output_edge = {};
  output_edge.FromNodeIndex = node_output->GetNode().index;
  output_edge.FromNodeOutputIndex = node_output->GetOutputIndex();
  uint32_t output_index = named_outputs_.size();
  output_edge.GraphOutputIndex = output_index;
  graph_desc_.output_edges.push_back(output_edge);

  named_outputs_[name] =
      node_output->GetTensorDesc().GetTotalTensorSizeInBytes();
}

ComPtr<IDMLCompiledOperator> GraphDescBuilder::Compile(
    DML_EXECUTION_FLAGS flags) {
  std::vector<DML_GRAPH_NODE_DESC> graph_nodes(graph_desc_.nodes.size());
  for (size_t i = 0; i < graph_nodes.size(); ++i) {
    graph_nodes[i] = {DML_GRAPH_NODE_TYPE_OPERATOR, &graph_desc_.nodes[i]};
  }

  std::vector<DML_GRAPH_EDGE_DESC> input_edges(graph_desc_.input_edges.size());
  for (size_t i = 0; i < input_edges.size(); ++i) {
    input_edges[i] = {DML_GRAPH_EDGE_TYPE_INPUT, &graph_desc_.input_edges[i]};
  }

  std::vector<DML_GRAPH_EDGE_DESC> output_edges(
      graph_desc_.output_edges.size());
  for (size_t i = 0; i < output_edges.size(); ++i) {
    output_edges[i] = {DML_GRAPH_EDGE_TYPE_OUTPUT,
                       &graph_desc_.output_edges[i]};
  }

  std::vector<DML_GRAPH_EDGE_DESC> intermediate_edges(
      graph_desc_.intermediate_edges.size());
  for (size_t i = 0; i < intermediate_edges.size(); ++i) {
    intermediate_edges[i] = {DML_GRAPH_EDGE_TYPE_INTERMEDIATE,
                             &graph_desc_.intermediate_edges[i]};
  }

  DML_GRAPH_DESC graph_desc = {};
  graph_desc.InputCount = input_nodes_.size();
  graph_desc.OutputCount = named_outputs_.size();
  graph_desc.NodeCount = static_cast<UINT>(graph_nodes.size());
  graph_desc.Nodes = graph_nodes.data();
  graph_desc.InputEdgeCount = static_cast<UINT>(input_edges.size());
  graph_desc.InputEdges = input_edges.data();
  graph_desc.OutputEdgeCount = static_cast<UINT>(output_edges.size());
  graph_desc.OutputEdges = output_edges.data();
  graph_desc.IntermediateEdgeCount =
      static_cast<UINT>(intermediate_edges.size());
  graph_desc.IntermediateEdges = intermediate_edges.data();

  ComPtr<IDMLDevice1> device1;
  HRESULT hr = device_->QueryInterface(IID_PPV_ARGS(&device1));
  if (FAILED(hr)) {
    return nullptr;
  }

  ComPtr<IDMLCompiledOperator> compiled_graph;
  hr = device1->CompileGraph(&graph_desc, flags, IID_PPV_ARGS(&compiled_graph));
  if (FAILED(hr)) {
    return nullptr;
  }
  return compiled_graph;
}

std::vector<InputNode>& GraphDescBuilder::GetInputNodes() {
  return input_nodes_;
}

std::map<std::string, size_t>& GraphDescBuilder::GetNamedOutputs() {
  return named_outputs_;
}

GraphDescBuilder::GraphDesc::GraphDesc() = default;
GraphDescBuilder::GraphDesc::~GraphDesc() = default;

}  // namespace content::webnn

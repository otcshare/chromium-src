// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/graph_node_output.h"

namespace content::webnn {

Node::Node(NodeType type, uint32_t index) {
  this->type = type;
  this->index = index;
}

Node::Node() = default;
Node::~Node() = default;

InputNode::InputNode() = default;
InputNode::~InputNode() = default;

OperatorNode::OperatorNode(const OperatorNode&& other) {
  this->op = std::move(other.op);
}
OperatorNode::OperatorNode() = default;
OperatorNode::~OperatorNode() = default;

NodeOutput::NodeOutput(Node node, uint32_t output_index, TensorDesc tensor_desc)
    : node_(node),
      output_index_(output_index),
      tensor_desc_(std::move(tensor_desc)) {}

NodeOutput::~NodeOutput() = default;

Node NodeOutput::GetNode() const {
  return node_;
}

uint32_t NodeOutput::GetOutputIndex() const {
  return output_index_;
}

TensorDesc& NodeOutput::GetTensorDesc() {
  return tensor_desc_;
}

}  // namespace content::webnn

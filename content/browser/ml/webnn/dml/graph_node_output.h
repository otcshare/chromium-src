// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_GRAPH_NODE_OUTPUT_H_
#define CONTENT_BROWSER_ML_WEBNN_GRAPH_NODE_OUTPUT_H_

#include <string>

#include <wrl.h>
#include "DirectML.h"
#include "content/browser/ml/webnn/dml/graph_tensor_desc.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;

enum class NodeType {
  kInput = 0,
  // Constant operand is also input for DirectML Graph builder.
  kConstant = 1,
  kOperator = 2,
  kUnknow = 3,
};

struct Node {
  Node();
  Node(NodeType type, uint32_t index);
  ~Node();

  NodeType type;
  // The index of this node in the Graph if it's output node which is counted
  // from 0. it also represent the index of input node in Graph's inputs which
  // is also counted from 0. Graph's inputs include constants and inputs of
  // execution.
  uint32_t index;
};

struct InputNode final : public Node {
  InputNode();
  ~InputNode();
  // The name identify input node which is to find memory info when binding
  // input for execution graph.
  std::string name;
  // The object id identify constant node which is to find memory info when
  // binding input for initializing graph.
  uint32_t object_id;
};

struct OperatorNode final : public Node {
  OperatorNode(const OperatorNode&& other);
  OperatorNode();
  ~OperatorNode();
  ComPtr<IDMLOperator> op;
};

class NodeOutput final {
 public:
  explicit NodeOutput(Node node,
                      uint32_t output_index,
                      TensorDesc tensor_desc);
  ~NodeOutput();

  Node GetNode() const;
  uint32_t GetOutputIndex() const;
  TensorDesc& GetTensorDesc();

 private:
  Node node_;
  // An operator can have multiple outputs. This index identifies which one of
  // the operator's outputs this node represents.
  uint32_t output_index_;
  // TODO::Wrapped into a class
  TensorDesc tensor_desc_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_GRAPH_NODE_OUTPUT_H_

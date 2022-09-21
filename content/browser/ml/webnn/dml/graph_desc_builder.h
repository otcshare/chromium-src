// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_GRAPH_DESC_BUILDER_H_
#define CONTENT_BROWSER_ML_WEBNN_GRAPH_DESC_BUILDER_H_

#include <string>
#include <vector>
#include <map>

#include "DirectML.h"
#include "content/browser/ml/webnn/dml/graph_node_output.h"

namespace content::webnn {

class GraphDescBuilder final {
 public:
  explicit GraphDescBuilder(ComPtr<IDMLDevice> device);
  ~GraphDescBuilder();

  Node CreateInputNode(std::string name);
  Node CreateConstantNode(uint32_t object_id);
  Node CreateOperatorNode(DML_OPERATOR_TYPE type, const void* desc);
  std::unique_ptr<NodeOutput> CreateNodeOutput(Node node,
                              uint32_t output_index,
                              DML_TENSOR_DESC tensor_desc);
  void Connect(std::vector<NodeOutput*>, Node operator_node);
  void AddOutputEdge(NodeOutput* node_output, std::string name);
  ComPtr<IDMLCompiledOperator> Compile(DML_EXECUTION_FLAGS flags);

  std::vector<InputNode>& GetInputNodes();
  std::map<std::string, size_t>& GetNamedOutputs();

 private:
  struct GraphDesc {
    GraphDesc();
    ~GraphDesc();
  
    std::vector<DML_OPERATOR_GRAPH_NODE_DESC> nodes;
    std::vector<DML_INPUT_GRAPH_EDGE_DESC> input_edges;
    std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> output_edges;
    std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediate_edges;
  };

  // The inputs node include inputs for execution and constant for
  // initialization because Both of them are inputs for DirectML Graph.
  std::vector<InputNode> input_nodes_;
  // The operator nodes hold a reference of IDMLOperator to be used for
  // GraphDesc.nodes
  std::vector<OperatorNode> operator_nodes_;
  // The output name and byte length mapping.
  std::map<std::string, size_t> named_outputs_;
  GraphDesc graph_desc_;
  ComPtr<IDMLDevice> device_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_GRAPH_DESC_BUILDER_H_

// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/execution_context.h"

#include "content/browser/ml/webnn/dml/adapter_dml.h"

namespace content::webnn {

ExecutionContext::ExecutionContext(scoped_refptr<AdapterDML> adapter)
    : command_queue_(adapter->GetCommandQueue()),
      command_recorder_(adapter->GetDMLDevice()) {}

ExecutionContext::~ExecutionContext() = default;

HRESULT ExecutionContext::Initialize() {
  HRESULT hr = command_recorder_.Initialize();
  if (FAILED(hr)) {
    return hr;
  }
  return S_OK;
}

HRESULT ExecutionContext::InitializeOperator(
    IDMLCompiledOperator* compiled_operator,
    const std::vector<std::shared_ptr<InputEdgeInfo>>& inputs) {
  return command_recorder_.InitializeOperator(compiled_operator, inputs);
}

HRESULT ExecutionContext::ExecuteOperator(
    IDMLCompiledOperator* compiled_operator,
    NamedInputsPtr namedInputs,
    const std::vector<std::shared_ptr<InputEdgeInfo>>& inputs,
    const std::vector<DML_BINDING_DESC>& output_bindings,
    UINT64 commonInputsResourceSize,
    ComPtr<ID3D12Resource> uploadResource,
    ComPtr<ID3D12Resource> inputResource) {
  return command_recorder_.ExecuteOperator(
      compiled_operator, std::move(namedInputs), inputs, output_bindings,
      commonInputsResourceSize, uploadResource, inputResource);
}

ComPtr<ID3D12CommandQueue> ExecutionContext::GetCommandQueue() {
  return command_queue_;
}

ComPtr<ID3D12CommandAllocator> ExecutionContext::GetCommandAllocator() {
  return command_recorder_.GetCommandAllocator();
}

ComPtr<ID3D12GraphicsCommandList> ExecutionContext::GetCommandList() {
  return command_recorder_.GetCommandList();
}

ComPtr<IDMLDevice> ExecutionContext::GetDMLDevice() {
  return command_recorder_.GetDMLDevice();
}

}  // namespace content::webnn

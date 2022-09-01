// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_

#include <wrl.h>
#include <vector>

#include "DirectML.h"
#include "components/ml/mojom/webnn_graph.mojom.h"
#include "content/browser/ml/webnn/dml/utils_dml.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
using ml::webnn::mojom::NamedInputsPtr;

class CommandRecorder final {
 public:
  explicit CommandRecorder(ComPtr<IDMLDevice> dml_device);
  ~CommandRecorder();

  CommandRecorder(const CommandRecorder&) = delete;
  CommandRecorder& operator=(const CommandRecorder&) = delete;

  HRESULT Initialize();

  HRESULT InitializeOperator(
      IDMLCompiledOperator* compiled_operator,
      const std::vector<std::shared_ptr<InputEdgeInfo>>& inputs);

  HRESULT ExecuteOperator(
      IDMLCompiledOperator* compiled_operator,
      NamedInputsPtr namedInputs,
      const std::vector<std::shared_ptr<InputEdgeInfo>>& inputs,
      const std::vector<DML_BINDING_DESC>& output_bindings,
      UINT64 commonInputsResourceSize,
      ComPtr<ID3D12Resource> uploadResource,
      ComPtr<ID3D12Resource> inputResource);

  // TODO:
  ComPtr<ID3D12CommandAllocator> GetCommandAllocator();
  ComPtr<ID3D12GraphicsCommandList> GetCommandList();
  ComPtr<IDMLDevice> GetDMLDevice();
  void FillUploadResourceAndInputBindings(
      UINT64 uploadResourceSize,
      std::vector<DML_BUFFER_BINDING>& inputBufferBinding,
      NamedInputsPtr namedInputs,
      const std::vector<std::shared_ptr<InputEdgeInfo>>& inputs,
      ComPtr<ID3D12Resource> uploadResource,
      ComPtr<ID3D12Resource> inputResource);

 private:
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<ID3D12Device> d3d12_device_;
  ComPtr<ID3D12CommandAllocator> command_allocator_;
  ComPtr<ID3D12GraphicsCommandList> command_list_;

  ComPtr<IDMLOperatorInitializer> operator_initializer_;
  ComPtr<IDMLCommandRecorder> command_recorder_;

  ComPtr<ID3D12DescriptorHeap> mDescriptorHeap;
  DML_BINDING_TABLE_DESC mBindingTableDesc;
  ComPtr<IDMLBindingTable> mBindingTable;

  ComPtr<ID3D12Resource> mUploadResource;
  ComPtr<ID3D12Resource> mInputResource;
  ComPtr<ID3D12Resource> mTemporaryResource;
  ComPtr<ID3D12Resource> mPersistentResource;

  UINT64 mTemporaryResourceSize = 0;
  UINT64 mPersistentResourceSize = 0;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_

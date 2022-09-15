// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_

#include <wrl.h>
#include <vector>

#include "DirectML.h"
#include "components/ml/mojom/webnn_graph.mojom.h"
#include "content/browser/ml/webnn/dml/gpgmm_d3d12.h"
#include "content/browser/ml/webnn/dml/utils_dml.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
using ml::webnn::mojom::NamedInputsPtr;

class AdapterDML;
class ExecutionResources;

class CommandRecorder final {
 public:
  explicit CommandRecorder(scoped_refptr<AdapterDML> adpter,
                           ComPtr<IDMLDevice> dml_device,
                           ComPtr<ID3D12CommandQueue> command_queue);
  ~CommandRecorder();

  CommandRecorder(const CommandRecorder&) = delete;
  CommandRecorder& operator=(const CommandRecorder&) = delete;

  HRESULT Initialize();

  void ResourceBarrier(std::vector<const D3D12_RESOURCE_BARRIER> barriers);

  void CopyBufferRegion(ID3D12Resource* dst_buffer,
                        uint64_t dst_offset,
                        ID3D12Resource* src_buffer,
                        uint64_t src_offset,
                        uint64_t byte_length);

  HRESULT InitializeGraph(uint32_t graph_id,
                          IDMLCompiledOperator* compiled_operator,
                          const DML_BINDING_DESC& input_array_binding);

  HRESULT ExecuteGraph(uint32_t graph_id,
                       IDMLCompiledOperator* compiled_operator,
                       const std::vector<DML_BINDING_DESC>& input_bindings,
                       const std::vector<DML_BINDING_DESC>& output_bindings);

  void CloseAndExecute();

  void SetExecutionResources(ExecutionResources* resources);

  // TODO:
  ComPtr<ID3D12CommandAllocator> GetCommandAllocator();
  ComPtr<ID3D12GraphicsCommandList> GetCommandList();
  ComPtr<IDMLDevice> GetDMLDevice();
  ComPtr<gpgmm::d3d12::ResourceAllocator> GetResourceAllocator();

 private:
  scoped_refptr<AdapterDML> adapter_;
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<ID3D12Device> d3d12_device_;
  ComPtr<ID3D12CommandAllocator> command_allocator_;
  ComPtr<ID3D12GraphicsCommandList> command_list_;
  ComPtr<ID3D12CommandQueue> command_queue_;

  ComPtr<IDMLOperatorInitializer> operator_initializer_;
  ComPtr<IDMLCommandRecorder> command_recorder_;

  ExecutionResources* unordered_resources_;

  ComPtr<gpgmm::d3d12::ResourceAllocator> resource_allocator_;
  ComPtr<ID3D12DescriptorHeap> mDescriptorHeap;
  DML_BINDING_TABLE_DESC mBindingTableDesc;
  ComPtr<IDMLBindingTable> mBindingTable;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_

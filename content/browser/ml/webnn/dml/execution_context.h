// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_EXECUTION_CONTEXT_H_
#define CONTENT_BROWSER_ML_WEBNN_EXECUTION_CONTEXT_H_

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>

#include "base/memory/ref_counted.h"
#include "content/browser/ml/webnn/dml/command_recorder.h"
#include "content/browser/ml/webnn/dml/execution_resources.h"
#include "content/browser/ml/webnn/dml/gpgmm_d3d12.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
class AdapterDML;
class ResourceAllocation;
class ResourceAllocatorManager;

class ExecutionContext final : public base::RefCounted<ExecutionContext> {
 public:
  explicit ExecutionContext(scoped_refptr<AdapterDML> adapter);

  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  HRESULT Initialize();

  void CopyBufferRegion(ID3D12Resource* dest_resource,
                        ID3D12Resource* src_resource,
                        UINT64 resource_size,
                        D3D12_RESOURCE_STATES state,
                        bool needBarrierEnd = true);

  HRESULT InitializeGraph(uint32_t graph_id,
                          IDMLCompiledOperator* compiled_operator,
                          const DML_BINDING_DESC& input_array_binding);

  HRESULT ExecuteGraph(uint32_t graph_id,
                       IDMLCompiledOperator* compiled_operator,
                       const std::vector<DML_BINDING_DESC>& input_bindings,
                       const std::vector<DML_BINDING_DESC>& output_bindings);

  // Forces all queued work to begin executing on the GPU.
  void Flush();

  ComPtr<ID3D12Device> GetD3D12Device() const;
  ComPtr<ID3D12CommandQueue> GetCommandQueue() const;
  ExecutionResources* GetExecutionResources();

  // ResourceAllocation AllocateMemory(
  //     D3D12_HEAP_TYPE heap_type,
  //     const D3D12_RESOURCE_DESC& resource_descriptor,
  //     D3D12_RESOURCE_STATES initial_usage);
  // void DeallocateMemory(ResourceAllocation& allocation);

  // TODO
  ComPtr<ID3D12CommandAllocator> GetCommandAllocator();
  ComPtr<ID3D12GraphicsCommandList> GetCommandList();
  ComPtr<IDMLDevice> GetDMLDevice();
  ComPtr<gpgmm::d3d12::ResourceAllocator> GetResourceAllocator();

 private:
  friend class base::RefCounted<ExecutionContext>;
  ~ExecutionContext();

  // Device is owned by adapter.
  ComPtr<ID3D12Device> d3d12_device_;
  // Open discussion: Another design is to share command queue for all contexts.
  ComPtr<ID3D12CommandQueue> command_queue_;
  // There is one active command recorder at a time.
  CommandRecorder command_recorder_;

  std::unique_ptr<ExecutionResources> unordered_resources_;
  // std::unique_ptr<ResourceAllocatorManager> resource_allocator_manager_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_EXECUTION_CONTEXT_H_

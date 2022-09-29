// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/readback_heap.h"

#include <memory>

#include "content/browser/ml/webnn/dml/execution_context.h"

namespace content::webnn {

ReadbackHeap::ReadbackHeap(ExecutionContext* execution_context)
    : execution_context_(execution_context), readback_resource_(nullptr) {}

ReadbackHeap::~ReadbackHeap() = default;

HRESULT ReadbackHeap::InitializeResource(
    std::map<std::string, size_t>& named_outputs) {
  uint64_t aligned_offset = 0;
  for (auto& [name, byte_length] : named_outputs) {
    MemoryInfo memory_info = {};
    memory_info.byte_offset = aligned_offset;
    memory_info.byte_length = byte_length;
    outputs_info_map_[name] = memory_info;

    // Only offset need to be algnement, the byte length keep original value.
    aligned_offset += Align(byte_length, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
  }
  outputs_resource_size_ = aligned_offset;
  outputs_shm_region_ =
      base::ReadOnlySharedMemoryRegion::Create(outputs_resource_size_);

  HRESULT hr = CreateReadbackResource(outputs_resource_size_);
  if (FAILED(hr)) {
    return hr;
  }
  return S_OK;
}

// Readback inference result from GPU that is stored in named_outputs.
HRESULT ReadbackHeap::ReadbackResource(NamedOutputsPtr& named_outputs,
                                       ID3D12Resource* src_resource) {
  // TODO:: Don't need add barrier to reset source resource from COPY_SOURCE
  // to UNORDERED_ACCESS?
  // Copy buffer from GPU resource to CPU data.
  execution_context_->CopyBufferRegion(readback_resource_->GetResource(),
                                       src_resource, outputs_resource_size_,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);

  execution_context_->Flush();

  D3D12_RANGE tensorBufferRange{0, outputs_resource_size_};
  int8_t* readBackBuffer;
  WEBNN_CHECK(readback_resource_->Map(
      0, &tensorBufferRange, reinterpret_cast<void**>(&readBackBuffer)));

  for (auto& [name, memory_info] : outputs_info_map_) {
    auto mojo_memory_info = ml::webnn::mojom::MemoryInfo::New();
    size_t byte_offset = memory_info.byte_offset;
    size_t byte_length = memory_info.byte_length;
    mojo_memory_info->byte_offset = byte_offset;
    mojo_memory_info->byte_length = byte_length;
    named_outputs->outputs[name] = std::move(mojo_memory_info);

    std::vector<uint8_t> output_buffer(byte_length);
    uint8_t* address =
        outputs_shm_region_.mapping.GetMemoryAs<uint8_t>() + byte_offset;
    memcpy(address, readBackBuffer + byte_offset, byte_length);
  }
  named_outputs->shared_memory = outputs_shm_region_.region.Duplicate();

  readback_resource_->Unmap(0, nullptr);
  return S_OK;
}

size_t ReadbackHeap::GetOutputsResourceSize() const {
  return outputs_resource_size_;
}

ReadbackHeap::MemoryInfo::MemoryInfo() = default;
ReadbackHeap::MemoryInfo::~MemoryInfo() = default;

HRESULT ReadbackHeap::CreateReadbackResource(size_t byte_length) {
  D3D12_HEAP_PROPERTIES heap_properties;
  // TODO::Support Unified Memory Architecture (UMA) that don't need to copy
  // anything there because GPU heaps are always mappable by CPU on unified,
  // D3D12_HEAP_TYPE_CUSTOM specify the memory pool and CPU cache properties
  // directly, which can be useful for UMA optimizations.
  heap_properties.Type = D3D12_HEAP_TYPE_READBACK;
  heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_properties.CreationNodeMask = 1;
  heap_properties.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC resource_desc;
  resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resource_desc.Alignment = 0;
  resource_desc.Width = byte_length;
  resource_desc.Height = 1;
  resource_desc.DepthOrArraySize = 1;
  resource_desc.MipLevels = 1;
  resource_desc.Format = DXGI_FORMAT_UNKNOWN;
  resource_desc.SampleDesc = {1, 0};
  resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resource_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

  gpgmm::d3d12::ALLOCATION_DESC allocation_descriptor = {};
  allocation_descriptor.HeapType = D3D12_HEAP_TYPE_READBACK;

  HRESULT hr = execution_context_->GetResourceAllocator()->CreateResource(
      allocation_descriptor, resource_desc, D3D12_RESOURCE_STATE_COPY_DEST,
      nullptr, &readback_resource_);

  if (FAILED(hr)) {
    return hr;
  }

  return S_OK;
}

}  // namespace content::webnn

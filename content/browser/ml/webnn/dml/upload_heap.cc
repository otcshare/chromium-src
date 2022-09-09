// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/upload_heap.h"

#include <memory>

#include "content/browser/ml/webnn/dml/execution_context.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::MemoryInfoPtr;

}  // namespace

UploadHeap::UploadHeap(ExecutionContext* execution_context)
    : execution_context_(execution_context) {}

UploadHeap::~UploadHeap() = default;

// The destination state represent the the state of destination resource that
// need to transition to
ID3D12Resource* UploadHeap::UploadResourceWithRingBuffer(
    ConstantsInfoPtr& constants_info) {
  base::ReadOnlySharedMemoryRegion& shared_memory_region =
      constants_info->shared_memory;
  size_t constants_byte_length = shared_memory_region.GetSize();
#ifdef ENABLE_GPU_MEMORY_MANAGEMENT
  // Allocate gpu resource with ResourceAllocator and manage it with Residency
  // management
#else
  // Use Committed resource directly that is managed by default.
  D3D12_RESOURCE_DESC resource_desc = CreateResourceDesc(
      constants_byte_length, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  constants_resource_ = CreateCommittedResource(
      execution_context_->GetD3D12Device().Get(), D3D12_HEAP_TYPE_DEFAULT,
      resource_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  if (constants_resource_ == nullptr) {
    return nullptr;
  }
#endif

  // TODO::
  ring_buffers_ =
      CreateRingBuffer(constants_byte_length);

  // Map the upload heap and copy the source data into it. A null pointer
  // indicates the entire subresource might be read by the CPU.
  void* upload_heap_data = nullptr;
  HRESULT hr = ring_buffers_->Map(0, nullptr, &upload_heap_data);
  if (FAILED(hr)) {
    return nullptr;
  }

  for (auto& [_, memory_info] : constants_info->constants) {
    // offset = RoundUpToMultiple(
    //     offset, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
    DCHECK(shared_memory_region.IsValid());
    base::ReadOnlySharedMemoryMapping shared_memory_mapping =
        shared_memory_region.MapAt(memory_info->byte_offset,
                                   memory_info->byte_length);
    memcpy(static_cast<byte*>(upload_heap_data) + memory_info->byte_offset,
           shared_memory_mapping.GetMemoryAs<uint8_t>(),
           memory_info->byte_length);
  }
  ring_buffers_->Unmap(0, nullptr);

  // Copy from the upload heap into the destination resource
  execution_context_->CopyBufferRegion(
      constants_resource_.Get(), ring_buffers_.Get(), constants_byte_length,
      D3D12_RESOURCE_STATE_COPY_DEST);

  return constants_resource_.Get();
}

// Reserve a ring buffer to accommodate the requested allocation size.
ComPtr<ID3D12Resource> UploadHeap::CreateRingBuffer(size_t src_byte_length) {
  D3D12_HEAP_PROPERTIES heap_properties;
  heap_properties.Type = D3D12_HEAP_TYPE_UPLOAD;
  heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_properties.CreationNodeMask = 1;
  heap_properties.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC resource_desc;
  resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resource_desc.Alignment = 0;
  resource_desc.Width = src_byte_length;
  resource_desc.Height = 1;
  resource_desc.DepthOrArraySize = 1;
  resource_desc.MipLevels = 1;
  resource_desc.Format = DXGI_FORMAT_UNKNOWN;
  resource_desc.SampleDesc = {1, 0};
  resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resource_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

  ComPtr<ID3D12Resource> upload_buffer;
  HRESULT hr = execution_context_->GetD3D12Device()->CreateCommittedResource(
      &heap_properties, D3D12_HEAP_FLAG_NONE, &resource_desc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload_buffer));
  if (FAILED(hr)) {
    return nullptr;
  }

  return upload_buffer;
}

}  // namespace content::webnn

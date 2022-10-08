// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/upload_heap.h"

#include <memory>

#include "content/browser/ml/webnn/dml/execution_context.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::MemoryInfoPtr;

template <typename T>
HRESULT UploadResource(ExecutionContext* execution_context,
                       ID3D12Resource* dst_resource,
                       ID3D12Resource* src_resource,
                       base::ReadOnlySharedMemoryRegion& shared_memory_region,
                       T& named_inputs) {
  // Map the upload heap and copy the source data into it. A null pointer
  // indicates the entire subresource might be read by the CPU.
  void* upload_heap_data = nullptr;
  HRESULT hr = src_resource->Map(0, nullptr, &upload_heap_data);
  if (FAILED(hr)) {
    return hr;
  }

  for (auto& [_, memory_info] : named_inputs) {
    uint64_t byte_length = memory_info->byte_length;
    uint64_t byte_offset = memory_info->byte_offset;
    DCHECK(byte_offset % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT == 0);
    DCHECK(shared_memory_region.IsValid());
    base::ReadOnlySharedMemoryMapping shared_memory_mapping =
        shared_memory_region.MapAt(memory_info->byte_offset, byte_length);
    memcpy(static_cast<byte*>(upload_heap_data) + memory_info->byte_offset,
           shared_memory_mapping.GetMemoryAs<uint8_t>(), byte_length);
  }
  src_resource->Unmap(0, nullptr);

  // Copy from the upload heap into the destination resource
  execution_context->CopyBufferRegion(dst_resource, src_resource,
                                      shared_memory_region.GetSize(),
                                      D3D12_RESOURCE_STATE_COPY_DEST);

  return S_OK;
}

}  // namespace

UploadHeap::UploadHeap(ExecutionContext* execution_context)
    : execution_context_(execution_context), upload_resource_(nullptr) {}

UploadHeap::~UploadHeap() = default;

// The destination state represent the the state of destination resource that
// need to transition.
HRESULT UploadHeap::UploadConstants(ID3D12Resource* dst_resource,
                                    ConstantsInfoPtr& constants_info) {
  base::ReadOnlySharedMemoryRegion& shared_memory_region =
      constants_info->shared_memory;
  size_t constants_byte_length = shared_memory_region.GetSize();

  HRESULT hr = S_OK;
  if (upload_resource_ == nullptr) {
    hr = CreateUploadResource(constants_byte_length);
    if (FAILED(hr)) {
      return hr;
    }
  }
  DCHECK(upload_resource_ != nullptr);

  return UploadResource<base::flat_map<uint32_t, MemoryInfoPtr>>(
      execution_context_, dst_resource, upload_resource_->GetResource(),
      shared_memory_region, constants_info->constants);
}

HRESULT UploadHeap::UploadInputs(ID3D12Resource* dst_resource,
                                 NamedInputsPtr& named_inputs) {
  base::ReadOnlySharedMemoryRegion& shared_memory_region =
      named_inputs->shared_memory;
  size_t inputs_byte_length = shared_memory_region.GetSize();

  HRESULT hr = S_OK;
  if (upload_resource_ == nullptr) {
    hr = CreateUploadResource(inputs_byte_length);
    if (FAILED(hr)) {
      return hr;
    }
  }
  DCHECK(upload_resource_ != nullptr);

  return UploadResource<base::flat_map<std::string, MemoryInfoPtr>>(
      execution_context_, dst_resource, upload_resource_->GetResource(),
      shared_memory_region, named_inputs->inputs);
}

// Create entire memory for uploading resource that will be uploaded piece by
// piece in GMM resource management.
HRESULT UploadHeap::CreateUploadResource(size_t byte_length) {
  D3D12_HEAP_PROPERTIES heap_properties;
  // TODO::Support Unified Memory Architecture (UMA) that don't need to copy
  // anything there because GPU heaps are always mappable by CPU on unified,
  // D3D12_HEAP_TYPE_CUSTOM specify the memory pool and CPU cache properties
  // directly, which can be useful for UMA optimizations.
  heap_properties.Type = D3D12_HEAP_TYPE_UPLOAD;
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
  allocation_descriptor.HeapType = D3D12_HEAP_TYPE_UPLOAD;

  HRESULT hr = execution_context_->GetResourceAllocator()->CreateResource(
      allocation_descriptor, resource_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
      nullptr, &upload_resource_);

  if (FAILED(hr)) {
    return hr;
  }

  return S_OK;
}

}  // namespace content::webnn

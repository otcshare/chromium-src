// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/execution_resources.h"

#include <memory>

#include "content/browser/ml/webnn/dml/execution_context.h"

namespace content::webnn {

ExecutionResources::ExecutionResources(ExecutionContext* execution_context)
    : execution_context_(execution_context) {}

ExecutionResources::~ExecutionResources() = default;

ComPtr<gpgmm::d3d12::ResourceAllocation> ExecutionResources::Allocate(
    UINT64 resource_size) {
  ID3D12Device* d3d12_device = execution_context_->GetD3D12Device().Get();
  // Use Committed resource directly that is managed by default.
  D3D12_HEAP_PROPERTIES heap_properties;
  heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
  heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_properties.CreationNodeMask = 1;
  heap_properties.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC resource_desc;
  resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resource_desc.Alignment = 0;
  resource_desc.Width = resource_size;
  resource_desc.Height = 1;
  resource_desc.DepthOrArraySize = 1;
  resource_desc.MipLevels = 1;
  resource_desc.Format = DXGI_FORMAT_UNKNOWN;
  resource_desc.SampleDesc = {1, 0};
  resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  // Check the resource size is valid, too large size could cause a device loss
  // when creating the resource.
  D3D12_RESOURCE_ALLOCATION_INFO resource_info =
      d3d12_device->GetResourceAllocationInfo(0, 1, &resource_desc);
  if (resource_info.SizeInBytes == 0 ||
      resource_info.SizeInBytes == std::numeric_limits<uint64_t>::max()) {
    // Invalid resource
    return nullptr;
  }

  ComPtr<gpgmm::d3d12::ResourceAllocation> resource;
  // TODO: Store a heap object for every allocated ResourceAllocation that will
  // be managed by residency management.
  gpgmm::d3d12::ALLOCATION_DESC allocation_descriptor = {};
  allocation_descriptor.HeapType = D3D12_HEAP_TYPE_DEFAULT;

  execution_context_->GetResourceAllocator()->CreateResource(
      allocation_descriptor, resource_desc,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &resource);

  return resource;
}

ID3D12Resource* ExecutionResources::Allocate(ResourceType type,
                                             UINT64 resource_size,
                                             UINT32 graph_id) {
  DCHECK_GT(graph_id, (uint32_t)(0));
  ComPtr<gpgmm::d3d12::ResourceAllocation> resource = Allocate(resource_size);
  if (pool_.find(graph_id) == pool_.end()) {
    pool_[graph_id] = Resources(type, resource);
  } else {
    auto& resources = pool_[graph_id].resources;
    resources[type] = resource;
  }
  return resource->GetResource();
}

ID3D12Resource* ExecutionResources::GetResource(UINT32 graph_id,
                                                ResourceType type) {
  auto iter = pool_.find(graph_id);
  if (iter == pool_.end()) {
    return nullptr;
  }

  auto& resources = iter->second.resources;
  if (resources.find(type) == resources.end()) {
    return nullptr;
  }
  return resources[type]->GetResource();
}

void ExecutionResources::Free(UINT32 graph_id) {
  auto iter = pool_.find(graph_id);
  if (iter != pool_.end()) {
    pool_.erase(iter);
  }
}

ExecutionResources::Resources::Resources() = default;
ExecutionResources::Resources::Resources(
    ResourceType type,
    ComPtr<gpgmm::d3d12::ResourceAllocation> resource) {
  resources[type] = resource;
}
ExecutionResources::Resources::~Resources() = default;

}  // namespace content::webnn

// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/unordered_resources.h"

#include <memory>

#include "content/browser/ml/webnn/dml/execution_context.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::MemoryInfoPtr;

}  // namespace

UnorderedResources::UnorderedResources(ExecutionContext* execution_context)
    : execution_context_(execution_context) {}

UnorderedResources::~UnorderedResources() = default;

ComPtr<ID3D12Resource> UnorderedResources::Allocate(UINT64 resource_size) {
#ifdef ENABLE_GPU_MEMORY_MANAGEMENT
  // Allocate gpu resource with ResourceAllocator and manage it with Residency
  // management
#else
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

  ComPtr<ID3D12Resource> resource;
  // D3D12 creates an implicit heap that contains the resource allocation when
  // calling CreateCommittedResource.
  // TODO: Store a heap object for every allocated ResourceAllocation that will
  // be managed by residency management.
  d3d12_device->CreateCommittedResource(
      &heap_properties, D3D12_HEAP_FLAG_NONE, &resource_desc,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&resource));
#endif

  return resource;
}

ID3D12Resource* UnorderedResources::Allocate(ResourceType type,
                                             UINT64 resource_size,
                                             UINT32 graph_id) {
  DCHECK_GT(graph_id, (uint32_t)(0));
  ComPtr<ID3D12Resource> resource = Allocate(resource_size);
  if (pool_.find(graph_id) == pool_.end()) {
    pool_[graph_id] = Resources(type, resource);
  } else {
    auto& resources = pool_[graph_id].resources;
    resources[type] = resource;
  }
  return resource.Get();
}

ID3D12Resource* UnorderedResources::GetResource(UINT32 graph_id,
                                                ResourceType type) {
  auto iter = pool_.find(graph_id);
  DCHECK(iter != pool_.end());
  auto& resources = iter->second.resources;
  DCHECK(resources.find(type) != resources.end());
  return resources[type].Get();
}

void UnorderedResources::Free(UINT32 graph_id) {
  auto iter = pool_.find(graph_id);
  if (iter != pool_.end()) {
    pool_.erase(iter);
  }
}

UnorderedResources::Resources::Resources() = default;
UnorderedResources::Resources::Resources(ResourceType type,
                                         ComPtr<ID3D12Resource> resource) {
  resources[type] = resource;
}
UnorderedResources::Resources::~Resources() = default;

}  // namespace content::webnn

// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_EXECUTION_RESOURCES_H_
#define CONTENT_BROWSER_ML_WEBNN_EXECUTION_RESOURCES_H_

#include <wrl.h>
#include <map>
#include <vector>

#include "DirectML.h"

#include "content/browser/ml/webnn/dml/gpgmm_d3d12.h"
#include "content/browser/ml/webnn/dml/utils_dml.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
class ExecutionContext;

enum class ResourceType {
  kInput = 0,
  kOutput = 1,
  kTemporary = 2,
  kPersistent = 3,
  kUnknow = 4,
};

// A unordered resources represent input, output, temporary and persistent
// resource.
class ExecutionResources final {
 public:
  explicit ExecutionResources(ExecutionContext* execution_context);
  ~ExecutionResources();

  // Allocate a resource that is released by manually.
  ComPtr<gpgmm::d3d12::ResourceAllocation> Allocate(UINT64 resource_size);
  // Allocate a resource that is owned by a graph and reused for execution.
  ID3D12Resource* Allocate(ResourceType type,
                           UINT64 resource_size,
                           UINT32 graph_id);
  ID3D12Resource* GetResource(UINT32 graph_id, ResourceType type);
  // Free a resource that is owned by graph such as temporary, persistent,
  // input and output unordered resouce for execution.
  void Free(UINT32 graph_id);

 private:
  struct Resources {
    Resources();
    Resources(ResourceType type, ComPtr<gpgmm::d3d12::ResourceAllocation> resource);
    ~Resources();

    std::map<ResourceType, ComPtr<gpgmm::d3d12::ResourceAllocation>> resources;
  };

  ExecutionContext* execution_context_;
  std::map<uint32_t, Resources> pool_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_EXECUTION_RESOURCES_H_

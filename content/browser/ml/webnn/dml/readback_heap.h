// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_READBACK_HEAP_H_
#define CONTENT_BROWSER_ML_WEBNN_READBACK_HEAP_H_

#include <wrl.h>
#include <vector>

#include "DirectML.h"
#include "components/ml/mojom/webnn_graph.mojom.h"
#include "content/browser/ml/webnn/dml/gpgmm_d3d12.h"
#include "content/browser/ml/webnn/dml/utils_dml.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
using ml::webnn::mojom::NamedOutputsPtr;

class ExecutionContext;

class ReadbackHeap final {
 public:
  explicit ReadbackHeap(ExecutionContext* execution_context);
  ~ReadbackHeap();

  HRESULT InitializeResource(std::map<std::string, size_t>& named_outputs);
  HRESULT ReadbackResource(NamedOutputsPtr& named_outputs,
                           ID3D12Resource* src_resource);
  size_t GetOutputsResourceSize() const;

 private:
  struct MemoryInfo {
    MemoryInfo();
    ~MemoryInfo();
    size_t byte_offset;
    size_t byte_length;
  };
  HRESULT CreateReadbackResource(size_t byte_length);

  ExecutionContext* execution_context_;
  UINT64 outputs_resource_size_ = 0;
  std::map<std::string, MemoryInfo> outputs_info_map_;
  base::MappedReadOnlyRegion outputs_shm_region_;
  ComPtr<gpgmm::d3d12::ResourceAllocation> readback_resource_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_READBACK_HEAP_H_

// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_UPLOAD_HEAP_H_
#define CONTENT_BROWSER_ML_WEBNN_UPLOAD_HEAP_H_

#include <wrl.h>
#include <vector>

#include "DirectML.h"
#include "components/ml/mojom/webnn_graph.mojom.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
using ml::webnn::mojom::ConstantsInfoPtr;
class ExecutionContext;

// A ring-buffer style upload heap for copying CPU data to GPU resources.
class UploadHeap final {
 public:
  explicit UploadHeap(ExecutionContext* execution_context);
  ~UploadHeap();

  HRESULT UploadConstants(ID3D12Resource* dst_resource,
                       ConstantsInfoPtr& constants_info);

 private:
  HRESULT CreateUploadResource(size_t byte_length);

  ExecutionContext* execution_context_;
#ifdef ENABLE_GPU_MEMORY_MANAGEMENT
  // GMM will re-use resource memory using FIFO for upload handle.
#else
  ComPtr<ID3D12Resource> upload_resource_;
#endif
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_UPLOAD_HEAP_H_

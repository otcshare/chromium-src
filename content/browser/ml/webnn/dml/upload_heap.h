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

  ID3D12Resource* UploadResourceWithRingBuffer(
    ConstantsInfoPtr& constants_info);

 private:
  ComPtr<ID3D12Resource> CreateRingBuffer(size_t src_byte_length);
  static constexpr uint64_t kRingBufferSize = 4 * 1024 * 1024;

  ExecutionContext* execution_context_;
  ComPtr<ID3D12Resource> ring_buffers_;
  // TODO::
  ComPtr<ID3D12Resource> constants_resource_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_UPLOAD_HEAP_H_

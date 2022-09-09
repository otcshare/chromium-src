// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_UPLOAD_HEAP_H_
#define CONTENT_BROWSER_ML_WEBNN_UPLOAD_HEAP_H_

#include <wrl.h>
#include <vector>

#include "DirectML.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
class ExecutionContext;

// A ring-buffer style upload heap for copying CPU data to GPU resources.
class UploadHeap final {
 public:
  explicit UploadHeap(ExecutionContext* execution_context);
  ~UploadHeap();

  void UploadResourceWithRingBuffer(ID3D12Resource* dst_resource,
                                    uint64_t dst_offset,
                                    void const* src_data,
                                    size_t src_byte_length);

 private:
  ComPtr<ID3D12Resource> CreateRingBuffer(size_t src_byte_length);
  static constexpr uint64_t kRingBufferSize = 4 * 1024 * 1024;

  ExecutionContext* execution_context_;
  std::vector<ComPtr<ID3D12Resource>> ring_buffers_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_UPLOAD_HEAP_H_

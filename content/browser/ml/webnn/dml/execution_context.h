// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_EXECUTION_CONTEXT_H_
#define CONTENT_BROWSER_ML_WEBNN_EXECUTION_CONTEXT_H_

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>

#include "base/memory/ref_counted.h"
#include "content/browser/ml/webnn/dml/command_recorder.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
class AdapterDML;

class ExecutionContext final : public base::RefCounted<ExecutionContext> {
 public:
  explicit ExecutionContext(scoped_refptr<AdapterDML> adapter);

  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  HRESULT Initialize();

  // TODO
  ComPtr<ID3D12CommandQueue> GetCommandQueue();
  ComPtr<ID3D12CommandAllocator> GetCommandAllocator();
  ComPtr<ID3D12GraphicsCommandList> GetCommandList();
  ComPtr<IDMLDevice> GetDMLDevice();

 private:
  friend class base::RefCounted<ExecutionContext>;
  ~ExecutionContext();

  // A shared command queue for all contexts that is owned by GPU process.
  ComPtr<ID3D12CommandQueue> command_queue_;
  // There is one active command recorder at a time.
  CommandRecorder command_recorder_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_EXECUTION_CONTEXT_H_

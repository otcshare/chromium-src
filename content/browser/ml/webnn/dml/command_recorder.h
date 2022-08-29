// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_

#include <wrl.h>

#include "DirectML.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;

class CommandRecorder final {
 public:
  explicit CommandRecorder(ComPtr<IDMLDevice> dml_device);
  ~CommandRecorder();

  CommandRecorder(const CommandRecorder&) = delete;
  CommandRecorder& operator=(const CommandRecorder&) = delete;

  HRESULT Initialize();

  //   void InitializeOperator(IDMLCompiledOperator* op,
  //                           const DML_BINDING_DESC&
  //                           persistentResourceBinding, const
  //                           DML_BINDING_DESC& inputArrayBinding);

  //   void ExecuteOperator(IDMLCompiledOperator* op,
  //                        const DML_BINDING_DESC& persistentResourceBinding,
  //                        std::span<const DML_BINDING_DESC> inputBindings,
  //                        std::span<const DML_BINDING_DESC> outputBindings);

  // TODO:
  ComPtr<ID3D12CommandAllocator> GetCommandAllocator();
  ComPtr<ID3D12GraphicsCommandList> GetCommandList();
  ComPtr<IDMLDevice> GetDMLDevice();

 private:
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<IDMLCommandRecorder> command_recorder_;
  ComPtr<ID3D12CommandAllocator> command_allocator_;
  ComPtr<ID3D12GraphicsCommandList> command_list_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_RECORDER_H_

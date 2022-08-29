// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/command_recorder.h"

namespace content::webnn {

CommandRecorder::~CommandRecorder() = default;

CommandRecorder::CommandRecorder(ComPtr<IDMLDevice> dml_device)
    : dml_device_(dml_device) {}

HRESULT CommandRecorder::Initialize() {
  ComPtr<ID3D12Device> d3d12_device;
  HRESULT hr = dml_device_->GetParentDevice(IID_PPV_ARGS(&d3d12_device));
  if (FAILED(hr)) {
    return hr;
  }

  hr = d3d12_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            IID_PPV_ARGS(&command_allocator_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = d3d12_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       command_allocator_.Get(), nullptr,
                                       IID_PPV_ARGS(&command_list_));
  if (FAILED(hr)) {
    return hr;
  }
  return S_OK;
}

ComPtr<ID3D12CommandAllocator> CommandRecorder::GetCommandAllocator() {
  return command_allocator_;
}

ComPtr<ID3D12GraphicsCommandList> CommandRecorder::GetCommandList() {
  return command_list_;
}

ComPtr<IDMLDevice> CommandRecorder::GetDMLDevice() {
  return dml_device_;
}

}  // namespace content::webnn

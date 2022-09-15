// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/adapter_dml.h"

namespace content::webnn {

AdapterDML::AdapterDML(ComPtr<IDXGIAdapter3> hardware_adapter)
    : hardware_adapter_(hardware_adapter) {}

AdapterDML::~AdapterDML() = default;

HRESULT AdapterDML::Initialize() {
  HRESULT hr =
      D3D12CreateDevice(hardware_adapter_.Get(), D3D_FEATURE_LEVEL_11_0,
                        IID_PPV_ARGS(&d3d12_device_));
  if (FAILED(hr)) {
    return hr;
  }

  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  command_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  hr = d3d12_device_->CreateCommandQueue(&command_queue_desc,
                                         IID_PPV_ARGS(&command_queue_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = DMLCreateDevice(d3d12_device_.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                       IID_PPV_ARGS(&dml_device_));
  if (FAILED(hr)) {
    return hr;
  }

  DXGI_ADAPTER_DESC1 adapter_desc;
  hardware_adapter_->GetDesc1(&adapter_desc);
  if (adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
    adapter_type_ = AdapterType::kCPU;
  } else {
    D3D12_FEATURE_DATA_ARCHITECTURE arch = {};
    hr = d3d12_device_->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch,
                                            sizeof(arch));
    if (FAILED(hr)) {
      return hr;
    }
    adapter_type_ =
        (arch.UMA) ? AdapterType::kIntegratedGPU : AdapterType::kDiscreteGPU;
  }

  return hr;
}

ComPtr<IDXGIAdapter3> AdapterDML::GetHardwareAdapter() const {
  DCHECK(hardware_adapter_.Get() != nullptr);
  return hardware_adapter_;
}

AdapterType AdapterDML::GetAdapterType() {
  DCHECK(adapter_type_ != AdapterType::kUnknow);
  return adapter_type_;
}

ComPtr<ID3D12Device> AdapterDML::GetD3D12Device() const {
  DCHECK(d3d12_device_.Get() != nullptr);
  return d3d12_device_;
}

ComPtr<IDMLDevice> AdapterDML::GetDMLDevice() const {
  DCHECK(dml_device_.Get() != nullptr);
  return dml_device_;
}

ComPtr<ID3D12CommandQueue> AdapterDML::GetCommandQueue() const {
  DCHECK(command_queue_.Get() != nullptr);
  return command_queue_;
}

}  // namespace content::webnn

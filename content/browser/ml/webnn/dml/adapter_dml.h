// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_ADAPTER_DML_H_
#define CONTENT_BROWSER_ML_WEBNN_ADAPTER_DML_H_

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>

#include "DirectML.h"
#include "base/memory/ref_counted.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;

// XPU can be extended in the adapter type.
enum class AdapterType {
  kDiscreteGPU = 0,
  kIntegratedGPU = 1,
  kCPU = 2,
  kUnknow = 3,
};

class AdapterDML final : public base::RefCounted<AdapterDML> {
 public:
  explicit AdapterDML(ComPtr<IDXGIAdapter3> hardware_adapter);

  AdapterDML(const AdapterDML&) = delete;
  AdapterDML& operator=(const AdapterDML&) = delete;

  HRESULT Initialize();
  ComPtr<IDXGIAdapter3> GetHardwareAdapter() const;
  AdapterType GetAdapterType();
  ComPtr<ID3D12Device> GetD3D12Device() const;
  ComPtr<IDMLDevice> GetDMLDevice() const;
  ComPtr<ID3D12CommandQueue> GetCommandQueue() const;

 private:
  friend class base::RefCounted<AdapterDML>;
  ~AdapterDML();

  ComPtr<IDXGIAdapter3> hardware_adapter_;
  AdapterType adapter_type_ = AdapterType::kUnknow;
  ComPtr<ID3D12Device> d3d12_device_;
  ComPtr<ID3D12CommandQueue> command_queue_;
  // Represents a DirectML device, which is used to create operators, binding
  // tables, command recorders.
  ComPtr<IDMLDevice> dml_device_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_ADAPTER_DML_H_

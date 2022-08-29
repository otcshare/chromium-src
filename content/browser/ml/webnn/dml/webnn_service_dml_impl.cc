// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/webnn_service_dml_impl.h"

#include <dxgi1_4.h>
#include <dxgi1_6.h>

#include "base/logging.h"
#include "base/no_destructor.h"
#include "content/browser/ml/webnn/dml/mojo_server_dml_impl.h"

namespace content::webnn {

namespace {

std::map<AdapterType, scoped_refptr<AdapterDML>> EumerateAdapters() {
  std::map<AdapterType, scoped_refptr<AdapterDML>> adapter_map = {};
  ComPtr<IDXGIFactory6> dxgi_factory = nullptr;
  HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));
  if (FAILED(hr)) {
    DLOG(ERROR) << "Create DXGI factory failed: "
                << logging::SystemErrorCodeToString(hr);
    return adapter_map;
  }

  // Eumerate all available adapters, DXGI_ERROR_NOT_FOUND means there are no
  // more adapters to enumerate.
  ComPtr<IDXGIAdapter1> dxgi_adapter;
  uint32_t adapter_index = 0;
  while (dxgi_factory->EnumAdapters1(adapter_index++, &dxgi_adapter) !=
         DXGI_ERROR_NOT_FOUND) {
    ComPtr<IDXGIAdapter3> dxgi_adapter3 = nullptr;
    hr = dxgi_adapter.As(&dxgi_adapter3);
    if (FAILED(hr)) {
      DLOG(ERROR) << "Get adapter3 failed: "
                  << logging::SystemErrorCodeToString(hr);
      return adapter_map;
    }
    auto adapter = base::MakeRefCounted<AdapterDML>(std::move(dxgi_adapter3));
    if (FAILED(adapter->Initialize())) {
      DLOG(ERROR) << "Initialize adapter failed: "
                  << logging::SystemErrorCodeToString(hr);
      return adapter_map;
    }

    adapter_map[adapter->GetAdapterType()] = adapter;
  }

  return adapter_map;
}

}  // namespace
// static
void WebnnServiceDMLImpl::Create(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver) {
  static base::NoDestructor<WebnnServiceDMLImpl> service{std::move(receiver)};
}

WebnnServiceDMLImpl::WebnnServiceDMLImpl(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver)
    : receiver_(this, std::move(receiver)), adapter_map_(EumerateAdapters()) {}

WebnnServiceDMLImpl::~WebnnServiceDMLImpl() = default;

void WebnnServiceDMLImpl::BindMojoServer(
    mojo::PendingReceiver<ml::webnn::mojom::MojoServer> receiver) {
  MojoServerDMLImpl::Create(std::move(receiver), this);
}

scoped_refptr<AdapterDML> WebnnServiceDMLImpl::RequestAdapter(
    PowerPreference power_preference) {
  AdapterType preferred_type;
  switch (power_preference) {
    case PowerPreference::kLowPower:
      preferred_type = AdapterType::kIntegratedGPU;
      break;
    case PowerPreference::kDefault:
    case PowerPreference::kHighPerformance:
      preferred_type = AdapterType::kDiscreteGPU;
      break;
  }
  auto iter = adapter_map_.find(preferred_type);
  if (iter != adapter_map_.end()) {
    return iter->second;
  }

  // Select device sequentially if there is no preferred type.
  iter = adapter_map_.find(AdapterType::kDiscreteGPU);
  if (iter != adapter_map_.end()) {
    return iter->second;
  }

  iter = adapter_map_.find(AdapterType::kIntegratedGPU);
  if (iter != adapter_map_.end()) {
    return iter->second;
  }

  iter = adapter_map_.find(AdapterType::kCPU);
  if (iter != adapter_map_.end()) {
    return iter->second;
  }

  return nullptr;
}

}  // namespace content::webnn

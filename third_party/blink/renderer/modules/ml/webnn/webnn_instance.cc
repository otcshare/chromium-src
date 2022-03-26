
// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_instance.h"

#include "gpu/command_buffer/client/webnn_interface.h"
#include "gpu/ipc/common/command_buffer_id.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/webgpu/context_provider.h"
#include "third_party/blink/renderer/platform/bindings/script_state.h"

namespace blink {

WebnnInstance::WebnnInstance() : webnn_control_client_(nullptr) {}

void WebnnInstance::Initialize(ExecutionContext* execution_context) {
  if (!webnn_control_client_) {
    // TODO(natlee@microsoft.com): if GPU process is lost, wait for the GPU
    // process to come back instead of rejecting right away
    std::unique_ptr<WebGraphicsContext3DProvider> context_provider =
        CreateContextProvider(*execution_context, gpu::CONTEXT_TYPE_WEBNN);
    if (!context_provider) {
      return;
    }
    // Make a new WebnnControlClientHolder with the context provider we just
    // made and set the lost context callback
    // TODO: Add kWebNN task type like TaskType::kWebGPU
    webnn_control_client_ = WebnnControlClientHolder::Create(
        std::move(context_provider),
        execution_context->GetTaskRunner(TaskType::kInternalDefault));
  }

  auto context_provider = webnn_control_client_->GetContextProviderWeakPtr();
  DCHECK(context_provider);
  gpu::webnn::WebNNInterface* webnn =
      context_provider->ContextProvider()->WebNNInterface();
  gpu::webnn::ReservedInstance reservation = webnn->ReserveInstance();
  webnn->InjectInstance(reservation.id, reservation.generation);
  instance_ = reservation.instance;

  return;
}

WNNContext WebnnInstance::CreateContext(MLContextOptions* options) {
  if (!webnn_control_client_ || webnn_control_client_->IsContextLost()) {
    return nullptr;
  }
  WNNContextOptions webnn_options = {};
  if (options->hasDevicePreference()) {
    webnn_options.devicePreference =
        options->devicePreference() == "cpu"   ? WNNDevicePreference_Cpu
        : options->devicePreference() == "gpu" ? WNNDevicePreference_Gpu
                                               : WNNDevicePreference_Default;
  }
  if (options->hasPowerPreference()) {
    webnn_options.powerPreference =
        options->powerPreference() == "high-performance"
            ? WNNPowerPreference_High_performance
        : options->devicePreference() == "low-power"
            ? WNNPowerPreference_Low_power
            : WNNPowerPreference_Default;
  }
  return webnn_control_client_->GetProcs().instanceCreateContext(
      instance_, &webnn_options);
}

WNNContext WebnnInstance::CreateContext(
    uint32_t device_id,
    uint32_t device_generation,
    gpu::CommandBufferId command_buffer_id) {
  if (!webnn_control_client_ || webnn_control_client_->IsContextLost()) {
    return nullptr;
  }

  auto context_provider = webnn_control_client_->GetContextProviderWeakPtr();
  DCHECK(context_provider);
  gpu::webnn::WebNNInterface* webnn =
      context_provider->ContextProvider()->WebNNInterface();
  webnn->InjectDawnWireServer(ChannelIdFromCommandBufferId(command_buffer_id),
                              RouteIdFromCommandBufferId(command_buffer_id));

  WNNGpuDevice webnn_device = {};
  webnn_device.id = device_id;
  webnn_device.generation = device_generation;
  return webnn_control_client_->GetProcs().instanceCreateContextWithGpuDevice(
      instance_, &webnn_device);
}

void WebnnInstance::ContextDestroyed() {
  if (webnn_control_client_) {
    webnn_control_client_->Destroy();
    webnn_control_client_ = nullptr;
  }
}

const scoped_refptr<WebnnControlClientHolder>&
WebnnInstance::GetWebnnControlClient() const {
  return webnn_control_client_;
}

WNNInstance WebnnInstance::GetInstance() const {
  return instance_;
}

}  // namespace blink

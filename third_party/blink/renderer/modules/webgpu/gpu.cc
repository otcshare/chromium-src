// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/webgpu/gpu.h"

#include <utility>

#include "base/compiler_specific.h"
#include "gpu/command_buffer/client/webgpu_interface.h"
#include "mojo/public/cpp/bindings/remote.h"
#include "third_party/blink/public/common/privacy_budget/identifiability_metric_builder.h"
#include "third_party/blink/public/common/privacy_budget/identifiability_study_settings.h"
#include "third_party/blink/public/common/privacy_budget/identifiable_token_builder.h"
#include "third_party/blink/public/platform/web_graphics_context_3d_provider.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_gpu_request_adapter_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/execution_context/navigator_base.h"
#include "third_party/blink/renderer/core/inspector/console_message.h"
#include "third_party/blink/renderer/modules/webgpu/context_provider.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_adapter.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_buffer.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_supported_features.h"
#include "third_party/blink/renderer/platform/graphics/gpu/dawn_control_client_holder.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"
#include "third_party/blink/renderer/platform/heap/thread_state.h"
#include "third_party/blink/renderer/platform/instrumentation/use_counter.h"
#include "third_party/blink/renderer/platform/privacy_budget/identifiability_digest_helpers.h"

namespace blink {

namespace {

ALLOW_UNUSED_TYPE void AddConsoleWarning(ExecutionContext* execution_context,
                                         const char* message) {
  if (execution_context) {
    auto* console_message = MakeGarbageCollected<ConsoleMessage>(
        mojom::blink::ConsoleMessageSource::kRendering,
        mojom::blink::ConsoleMessageLevel::kWarning, message);
    execution_context->AddConsoleMessage(console_message);
  }
}

}  // anonymous namespace

// static
const char GPU::kSupplementName[] = "GPU";

// static
GPU* GPU::gpu(NavigatorBase& navigator) {
  GPU* gpu = Supplement<NavigatorBase>::From<GPU>(navigator);
  if (!gpu) {
    gpu = MakeGarbageCollected<GPU>(navigator);
    ProvideTo(navigator, gpu);
  }
  return gpu;
}

GPU::GPU(NavigatorBase& navigator)
    : Supplement<NavigatorBase>(navigator),
      ExecutionContextLifecycleObserver(navigator.GetExecutionContext()) {}

GPU::~GPU() = default;

void GPU::Trace(Visitor* visitor) const {
  ScriptWrappable::Trace(visitor);
  Supplement<NavigatorBase>::Trace(visitor);
  ExecutionContextLifecycleObserver::Trace(visitor);
  visitor->Trace(mappable_buffers_);
}

void GPU::ContextDestroyed() {
  if (!mappable_buffers_.IsEmpty()) {
    // Destroy all mappable buffers. This ensures all mappings backed by
    // shared memory are detached before the WebGPU command buffer and
    // transfer buffers are destroyed.
    v8::Isolate* isolate = ThreadState::Current()->GetIsolate();
    for (GPUBuffer* buffer : mappable_buffers_) {
      buffer->Destroy(isolate);
    }
  }
  if (!dawn_control_client_) {
    return;
  }
  dawn_control_client_->Destroy();
}

void GPU::OnRequestAdapterCallback(ScriptState* script_state,
                                   const GPURequestAdapterOptions* options,
                                   ScriptPromiseResolver* resolver,
                                   int32_t adapter_server_id,
                                   const WGPUDeviceProperties& properties,
                                   const char* error_message) {
  GPUAdapter* adapter = nullptr;
  if (adapter_server_id >= 0) {
    adapter = MakeGarbageCollected<GPUAdapter>(
        this, "Default", adapter_server_id, properties, dawn_control_client_);
  }
  if (error_message) {
    ExecutionContext* execution_context = ExecutionContext::From(script_state);
    auto* console_message = MakeGarbageCollected<ConsoleMessage>(
        mojom::blink::ConsoleMessageSource::kRendering,
        mojom::blink::ConsoleMessageLevel::kWarning, error_message);
    execution_context->AddConsoleMessage(console_message);
  }
  RecordAdapterForIdentifiability(script_state, options, adapter);
  resolver->Resolve(adapter);
}

void GPU::RecordAdapterForIdentifiability(
    ScriptState* script_state,
    const GPURequestAdapterOptions* options,
    GPUAdapter* adapter) const {
  constexpr IdentifiableSurface::Type type =
      IdentifiableSurface::Type::kGPU_RequestAdapter;
  if (!IdentifiabilityStudySettings::Get()->ShouldSample(type))
    return;
  ExecutionContext* context = GetExecutionContext();
  if (!context)
    return;

  IdentifiableTokenBuilder input_builder;
  if (options && options->hasPowerPreference()) {
    input_builder.AddToken(
        IdentifiabilityBenignStringToken(options->powerPreference()));
  }
  const auto surface =
      IdentifiableSurface::FromTypeAndToken(type, input_builder.GetToken());

  IdentifiableTokenBuilder output_builder;
  if (adapter) {
    output_builder.AddToken(IdentifiabilityBenignStringToken(adapter->name()));
    for (const auto& feature : adapter->features()->FeatureNameSet()) {
      output_builder.AddToken(IdentifiabilityBenignStringToken(feature));
    }
  }

  IdentifiabilityMetricBuilder(context->UkmSourceID())
      .Add(surface, output_builder.GetToken())
      .Record(context->UkmRecorder());
}

ScriptPromise GPU::requestAdapter(ScriptState* script_state,
                                  const GPURequestAdapterOptions* options) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();

  if (!dawn_control_client_ || dawn_control_client_->IsContextLost()) {
    ExecutionContext* execution_context = ExecutionContext::From(script_state);

    // TODO(natlee@microsoft.com): if GPU process is lost, wait for the GPU
    // process to come back instead of rejecting right away
    std::unique_ptr<WebGraphicsContext3DProvider> context_provider =
        CreateContextProvider(*execution_context);

    if (!context_provider) {
      // Failed to create context provider, won't be able to request adapter
      // TODO(crbug.com/973017): Collect GPU info and surface context creation
      // error.
      resolver->Resolve(v8::Null(script_state->GetIsolate()));
      return promise;
    } else {
      // Make a new DawnControlClientHolder with the context provider we just
      // made and set the lost context callback
      dawn_control_client_ = DawnControlClientHolder::Create(
          std::move(context_provider),
          execution_context->GetTaskRunner(TaskType::kWebGPU));
    }
  }

  // For now we choose kHighPerformance by default.
  gpu::webgpu::PowerPreference power_preference =
      gpu::webgpu::PowerPreference::kHighPerformance;
  if (options->hasPowerPreference() &&
      options->powerPreference() == "low-power") {
    power_preference = gpu::webgpu::PowerPreference::kLowPower;
  }

  auto context_provider = dawn_control_client_->GetContextProviderWeakPtr();
  DCHECK(context_provider);
  context_provider->ContextProvider()->WebGPUInterface()->RequestAdapterAsync(
      power_preference, options->forceFallbackAdapter(),
      WTF::Bind(&GPU::OnRequestAdapterCallback, WrapPersistent(this),
                WrapPersistent(script_state), WrapPersistent(options),
                WrapPersistent(resolver)));

  UseCounter::Count(ExecutionContext::From(script_state), WebFeature::kWebGPU);

  return promise;
}

void GPU::TrackMappableBuffer(GPUBuffer* buffer) {
  mappable_buffers_.insert(buffer);
}

}  // namespace blink

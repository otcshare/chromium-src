// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_CONTEXT_PROVIDER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_CONTEXT_PROVIDER_H_

#include "base/synchronization/waitable_event.h"
#include "base/task/single_thread_task_runner.h"
#include "gpu/command_buffer/common/context_creation_attribs.h"
#include "third_party/blink/public/common/thread_safe_browser_interface_broker_proxy.h"
#include "third_party/blink/public/mojom/gpu/gpu.mojom-blink.h"
#include "third_party/blink/public/platform/platform.h"
#include "third_party/blink/public/platform/web_graphics_context_3d_provider.h"
#include "third_party/blink/renderer/platform/scheduler/public/post_cross_thread_task.h"
#include "third_party/blink/renderer/platform/scheduler/public/thread.h"
#include "third_party/blink/renderer/platform/weborigin/kurl.h"
#include "third_party/blink/renderer/platform/wtf/cross_thread_functional.h"

namespace blink {

namespace {

void CreateContextProvider(
    const KURL& url,
    gpu::ContextType context_type,
    base::WaitableEvent* waitable_event,
    std::unique_ptr<WebGraphicsContext3DProvider>* created_context_provider) {
  DCHECK(IsMainThread());
  *created_context_provider =
      context_type == gpu::CONTEXT_TYPE_WEBGPU
          ? Platform::Current()->CreateWebGPUGraphicsContext3DProvider(url)
          : Platform::Current()->CreateWebNNContextProvider(url);
  waitable_event->Signal();
}

std::unique_ptr<WebGraphicsContext3DProvider> CreateContextProviderOnMainThread(
    const KURL& url,
    gpu::ContextType context_type) {
  scoped_refptr<base::SingleThreadTaskRunner> task_runner =
      Thread::MainThread()->GetTaskRunner();

  base::WaitableEvent waitable_event;
  std::unique_ptr<WebGraphicsContext3DProvider> created_context_provider;
  PostCrossThreadTask(
      *task_runner, FROM_HERE,
      CrossThreadBindOnce(&CreateContextProvider, url, context_type,
                          CrossThreadUnretained(&waitable_event),
                          CrossThreadUnretained(&created_context_provider)));

  waitable_event.Wait();
  return created_context_provider;
}

}  // namespace

std::unique_ptr<WebGraphicsContext3DProvider> inline CreateContextProvider(
    ExecutionContext& execution_context,
    gpu::ContextType context_type = gpu::CONTEXT_TYPE_WEBGPU) {
  const KURL& url = execution_context.Url();
  std::unique_ptr<WebGraphicsContext3DProvider> context_provider;
  if (IsMainThread()) {
    context_provider =
        context_type == gpu::CONTEXT_TYPE_WEBGPU
            ? Platform::Current()->CreateWebGPUGraphicsContext3DProvider(url)
            : Platform::Current()->CreateWebNNContextProvider(url);
  } else {
    context_provider = CreateContextProviderOnMainThread(url, context_type);
  }

  // Note that we check for API blocking *after* creating the context. This is
  // because context creation synchronizes against GpuProcessHost lifetime in
  // the browser process, and GpuProcessHost destruction is what updates API
  // blocking state on a GPU process crash. See https://crbug.com/1215907#c10
  // for more details.
  bool blocked = true;
  mojo::Remote<mojom::blink::GpuDataManager> gpu_data_manager;
  Platform::Current()->GetBrowserInterfaceBroker()->GetInterface(
      gpu_data_manager.BindNewPipeAndPassReceiver());
  gpu_data_manager->Are3DAPIsBlockedForUrl(url, &blocked);
  if (blocked) {
    return nullptr;
  }

  // TODO(kainino): we will need a better way of accessing the GPU interface
  // from multiple threads than BindToCurrentThread et al.
  if (context_provider && !context_provider->BindToCurrentThread()) {
    // TODO(crbug.com/973017): Collect GPU info and surface context creation
    // error.
    return nullptr;
  }
  return context_provider;
}

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_CONTEXT_PROVIDER_H_
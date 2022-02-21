// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_GPU_WEBNN_CONTROL_CLIENT_HOLDER_H_
#define THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_GPU_WEBNN_CONTROL_CLIENT_HOLDER_H_

#include <webnn/webnn_proc_table.h>
#include <webnn/webnn.h>

#include "gpu/command_buffer/client/webnn_interface.h"
#include "third_party/blink/renderer/platform/graphics/web_graphics_context_3d_provider_wrapper.h"
#include "third_party/blink/renderer/platform/platform_export.h"
#include "third_party/blink/renderer/platform/wtf/ref_counted.h"

namespace base {

class SingleThreadTaskRunner;

}  // namespace base

namespace blink {

// This class holds the WebGraphicsContext3DProviderWrapper and a strong
// reference to the WebNN APIChannel.
// WebnnControlClientHolder::Destroy() should be called to destroy the
// context which will free all command buffer and GPU resources. As long
// as the reference to the APIChannel is held, calling WebNN procs is
// valid.
class PLATFORM_EXPORT WebnnControlClientHolder
    : public RefCounted<WebnnControlClientHolder> {
 public:
  static scoped_refptr<WebnnControlClientHolder> Create(
      std::unique_ptr<WebGraphicsContext3DProvider> context_provider,
      scoped_refptr<base::SingleThreadTaskRunner> task_runner);

  WebnnControlClientHolder(
      std::unique_ptr<WebGraphicsContext3DProvider> context_provider,
      scoped_refptr<base::SingleThreadTaskRunner> task_runner);

  void Destroy();

  // Returns a weak pointer to |context_provider_|. If the pointer is valid and
  // non-null, the WebNN context has not been destroyed, and it is safe to use
  // the WebNN interface.
  base::WeakPtr<WebGraphicsContext3DProviderWrapper> GetContextProviderWeakPtr()
      const;
  const WebnnProcTable& GetProcs() const { return procs_; }
  bool IsContextLost() const;

 private:
  friend class RefCounted<WebnnControlClientHolder>;
  ~WebnnControlClientHolder();

  std::unique_ptr<WebGraphicsContext3DProviderWrapper> context_provider_;
  scoped_refptr<base::SingleThreadTaskRunner> task_runner_;
  scoped_refptr<gpu::webnn::APIChannel> api_channel_;
  WebnnProcTable procs_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_GPU_WEBNN_CONTROL_CLIENT_HOLDER_H_

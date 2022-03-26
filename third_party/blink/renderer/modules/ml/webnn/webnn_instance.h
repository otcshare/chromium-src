// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_INSTANCE_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_INSTANCE_H_

#include "base/memory/scoped_refptr.h"
#include "gpu/command_buffer/common/command_buffer_id.h"
#include "third_party/blink/renderer/platform/graphics/gpu/webnn_control_client_holder.h"

namespace blink {

class MLContextOptions;
class ScriptState;
class ExecutionContext;

class WebnnInstance final {
 public:
  WebnnInstance();

  WebnnInstance(const WebnnInstance&) = delete;
  WebnnInstance& operator=(const WebnnInstance&) = delete;

  void Initialize(ExecutionContext* execution_context);
  WNNContext CreateContext(MLContextOptions* options);
  WNNContext CreateContext(uint32_t device_id,
                           uint32_t device_generation,
                           gpu::CommandBufferId command_buffer_id);
  void ContextDestroyed();

  const scoped_refptr<WebnnControlClientHolder>& GetWebnnControlClient() const;
  WNNInstance GetInstance() const;

 private:
  scoped_refptr<WebnnControlClientHolder> webnn_control_client_;
  WNNInstance instance_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_INSTANCE_H_

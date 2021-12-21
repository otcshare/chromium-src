// Copyright (c) 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_SERVICE_WEBNN_DECODER_H_
#define GPU_COMMAND_BUFFER_SERVICE_WEBNN_DECODER_H_

#include "gpu/command_buffer/service/common_decoder.h"
#include "gpu/command_buffer/service/decoder_context.h"
#include "gpu/gpu_gles2_export.h"

namespace gpu {

class DecoderClient;
struct GpuPreferences;
class MemoryTracker;
class SharedImageManager;

namespace webnn {

class GPU_GLES2_EXPORT WebNNDecoder : public DecoderContext,
                                      public CommonDecoder {
 public:
  static WebNNDecoder* Create(DecoderClient* client,
                              CommandBufferServiceBase* command_buffer_service,
                              SharedImageManager* shared_image_manager,
                              MemoryTracker* memory_tracker,
                              const GpuPreferences& gpu_preferences);

  WebNNDecoder(const WebNNDecoder&) = delete;
  WebNNDecoder& operator=(const WebNNDecoder&) = delete;

  ~WebNNDecoder() override;

  // WebNN-specific initialization that's different than DecoderContext's
  // Initialize that is tied to GLES2 concepts and a noop for WebNN decoders.
  virtual ContextResult Initialize() = 0;

  ContextResult Initialize(const scoped_refptr<gl::GLSurface>& surface,
                           const scoped_refptr<gl::GLContext>& context,
                           bool offscreen,
                           const gles2::DisallowedFeatures& disallowed_features,
                           const ContextCreationAttribs& attrib_helper) final;

 protected:
  WebNNDecoder(DecoderClient* client,
               CommandBufferServiceBase* command_buffer_service);
};

}  // namespace webnn
}  // namespace gpu

#endif  // GPU_COMMAND_BUFFER_SERVICE_WEBNN_DECODER_H_

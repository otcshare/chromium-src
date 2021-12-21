// Copyright (c) 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "gpu/command_buffer/service/webnn_decoder.h"

#include "gpu/command_buffer/service/webnn_decoder_impl.h"
#include "ui/gl/buildflags.h"

namespace gpu {
namespace webnn {

// static
WebNNDecoder* WebNNDecoder::Create(
    DecoderClient* client,
    CommandBufferServiceBase* command_buffer_service,
    SharedImageManager* shared_image_manager,
    MemoryTracker* memory_tracker,
    const GpuPreferences& gpu_preferences) {
  return CreateWebNNDecoderImpl(client, command_buffer_service,
                                shared_image_manager, memory_tracker,
                                gpu_preferences);
}

WebNNDecoder::WebNNDecoder(DecoderClient* client,
                           CommandBufferServiceBase* command_buffer_service)
    : CommonDecoder(client, command_buffer_service) {}

WebNNDecoder::~WebNNDecoder() {}

ContextResult WebNNDecoder::Initialize(
    const scoped_refptr<gl::GLSurface>& surface,
    const scoped_refptr<gl::GLContext>& context,
    bool offscreen,
    const gles2::DisallowedFeatures& disallowed_features,
    const ContextCreationAttribs& attrib_helper) {
  return ContextResult::kSuccess;
}

}  // namespace webnn
}  // namespace gpu

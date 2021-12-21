// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_SERVICE_WEBNN_DECODER_IMPL_H_
#define GPU_COMMAND_BUFFER_SERVICE_WEBNN_DECODER_IMPL_H_

#include "gpu/gpu_gles2_export.h"

namespace gpu {

class CommandBufferServiceBase;
class DecoderClient;
struct GpuPreferences;
class MemoryTracker;
class SharedImageManager;

namespace webnn {

class WebNNDecoder;

GPU_GLES2_EXPORT WebNNDecoder* CreateWebNNDecoderImpl(
    DecoderClient* client,
    CommandBufferServiceBase* command_buffer_service,
    SharedImageManager* shared_image_manager,
    MemoryTracker* memory_tracker,
    const GpuPreferences& gpu_preferences);

}  // namespace webnn
}  // namespace gpu

#endif  // GPU_COMMAND_BUFFER_SERVICE_WEBNN_DECODER_IMPL_H_

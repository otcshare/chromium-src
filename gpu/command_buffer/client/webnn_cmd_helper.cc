// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "gpu/command_buffer/client/webnn_cmd_helper.h"

namespace gpu {
namespace webnn {

WebNNCmdHelper::WebNNCmdHelper(CommandBuffer* command_buffer)
    : CommandBufferHelper(command_buffer) {}

WebNNCmdHelper::~WebNNCmdHelper() = default;

}  // namespace webnn
}  // namespace gpu

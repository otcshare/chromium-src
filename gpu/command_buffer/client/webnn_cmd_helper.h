// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_CLIENT_WEBNN_CMD_HELPER_H_
#define GPU_COMMAND_BUFFER_CLIENT_WEBNN_CMD_HELPER_H_

#include <stdint.h>

#include "gpu/command_buffer/client/cmd_buffer_helper.h"
#include "gpu/command_buffer/client/webnn_export.h"
#include "gpu/command_buffer/common/webnn_cmd_format.h"

namespace gpu {
namespace webnn {

// A class that helps write WebNN command buffers.
class WEBNN_IMPLEMENTATION_EXPORT WebNNCmdHelper : public CommandBufferHelper {
 public:
  explicit WebNNCmdHelper(CommandBuffer* command_buffer);

  WebNNCmdHelper(const WebNNCmdHelper&) = delete;
  WebNNCmdHelper& operator=(const WebNNCmdHelper&) = delete;

  ~WebNNCmdHelper() override;

// Include the auto-generated part of this class. We split this because it
// means we can easily edit the non-auto generated parts right here in this
// file instead of having to edit some template or the code generator.
#include "gpu/command_buffer/client/webnn_cmd_helper_autogen.h"
};

}  // namespace webnn
}  // namespace gpu

#endif  // GPU_COMMAND_BUFFER_CLIENT_WEBNN_CMD_HELPER_H_

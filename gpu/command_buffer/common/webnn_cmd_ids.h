// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file defines the webnn command buffer commands.

#ifndef GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_IDS_H_
#define GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_IDS_H_

#include "gpu/command_buffer/common/cmd_buffer_common.h"

namespace gpu {
namespace webnn {

#include "gpu/command_buffer/common/webnn_cmd_ids_autogen.h"

const char* GetCommandName(CommandId command_id);

}  // namespace webnn
}  // namespace gpu

#endif  // GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_IDS_H_

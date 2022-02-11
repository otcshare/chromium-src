// Copyright (c) 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_H_
#define GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_H_

#include <string.h>

#include "gpu/command_buffer/common/gl2_types.h"
#include "gpu/command_buffer/common/gles2_cmd_utils.h"
#include "gpu/command_buffer/common/webnn_cmd_enums.h"
#include "gpu/command_buffer/common/webnn_cmd_ids.h"

namespace gpu {
namespace webnn {
namespace cmds {

#define GPU_WEBNN_RETURN_DATA_ALIGNMENT (8)
struct alignas(GPU_WEBNN_RETURN_DATA_ALIGNMENT) WebnnReturnDataHeader {
  WebnnReturnDataType return_data_type;
};

static_assert(
    sizeof(WebnnReturnDataHeader) % GPU_WEBNN_RETURN_DATA_ALIGNMENT == 0,
    "WebnnReturnDataHeader must align to GPU_WEBNN_RETURN_DATA_ALIGNMENT");

struct WebnnReturnCommandsInfoHeader {
  WebnnReturnDataHeader return_data_header = {WebnnReturnDataType::kWebnnCommands};
};

static_assert(offsetof(WebnnReturnCommandsInfoHeader, return_data_header) == 0,
              "The offset of return_data_header must be 0");

struct WebnnReturnCommandsInfo {
  WebnnReturnCommandsInfoHeader header;
  alignas(GPU_WEBNN_RETURN_DATA_ALIGNMENT) char deserialized_buffer[];
};

static_assert(offsetof(WebnnReturnCommandsInfo, header) == 0,
              "The offset of header must be 0");


// Command buffer is GPU_COMMAND_BUFFER_ENTRY_ALIGNMENT byte aligned.
#pragma pack(push, 4)
static_assert(GPU_COMMAND_BUFFER_ENTRY_ALIGNMENT == 4,
              "pragma pack alignment must be equal to "
              "GPU_COMMAND_BUFFER_ENTRY_ALIGNMENT");

#include "gpu/command_buffer/common/webnn_cmd_format_autogen.h"

#pragma pack(pop)

}  // namespace cmds
}  // namespace webnn
}  // namespace gpu
#endif  // GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_H_

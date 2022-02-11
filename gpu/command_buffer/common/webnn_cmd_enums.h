// Copyright (c) 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_ENUMS_H_
#define GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_ENUMS_H_

#include <stdint.h>

namespace gpu {
namespace webnn {

enum class WebnnReturnDataType : uint32_t {
  kWebnnCommands,
  kNumWebnnReturnDataType
};

enum MailboxFlags : uint32_t {
  WEBNN_MAILBOX_NONE = 0,
  WEBNN_MAILBOX_DISCARD = 1 << 0,
};

}  // namespace webnn
}  // namespace gpu
#endif  // GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_ENUMS_H_

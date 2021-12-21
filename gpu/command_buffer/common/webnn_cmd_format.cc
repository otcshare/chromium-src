// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file contains the binary format definition of the command buffer and
// command buffer commands.

// We explicitly do NOT include webnn_cmd_format.h here because client side
// and service side have different requirements.
#include "base/cxx17_backports.h"
#include "gpu/command_buffer/common/cmd_buffer_common.h"

namespace gpu {
namespace webnn {

#include "gpu/command_buffer/common/webnn_cmd_ids_autogen.h"

const char* GetCommandName(CommandId id) {
  static const char* const names[] = {
#define WEBNN_CMD_OP(name) "k" #name,

      WEBNN_COMMAND_LIST(WEBNN_CMD_OP)

#undef WEBNN_CMD_OP
  };

  size_t index = static_cast<size_t>(id) - kFirstWebNNCommand;
  return (index < std::size(names)) ? names[index] : "*unknown-command*";
}

}  // namespace webnn
}  // namespace gpu

// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file is auto-generated from
// gpu/command_buffer/build_webnn_cmd_buffer.py
// It's formatted by clang-format using chromium coding style:
//    clang-format -i -style=chromium filename
// DO NOT EDIT!

#ifndef GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_IDS_AUTOGEN_H_
#define GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_IDS_AUTOGEN_H_

#define WEBNN_COMMAND_LIST(OP)       \
  OP(WebnnCommands)        /* 256 */ \
  OP(InjectInstance)       /* 257 */ \
  OP(InjectDawnWireServer) /* 258 */

enum CommandId {
  kOneBeforeStartPoint =
      cmd::kLastCommonId,  // All WebNN commands start after this.
#define WEBNN_CMD_OP(name) k##name,
  WEBNN_COMMAND_LIST(WEBNN_CMD_OP)
#undef WEBNN_CMD_OP
      kNumCommands,
  kFirstWebNNCommand = kOneBeforeStartPoint + 1
};

#endif  // GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_IDS_AUTOGEN_H_

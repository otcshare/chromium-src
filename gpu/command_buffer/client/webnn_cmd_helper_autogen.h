// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file is auto-generated from
// gpu/command_buffer/build_webnn_cmd_buffer.py
// It's formatted by clang-format using chromium coding style:
//    clang-format -i -style=chromium filename
// DO NOT EDIT!

#ifndef GPU_COMMAND_BUFFER_CLIENT_WEBNN_CMD_HELPER_AUTOGEN_H_
#define GPU_COMMAND_BUFFER_CLIENT_WEBNN_CMD_HELPER_AUTOGEN_H_

void WebnnCommands(uint32_t commands_shm_id,
                   uint32_t commands_shm_offset,
                   uint32_t size) {
  webnn::cmds::WebnnCommands* c = GetCmdSpace<webnn::cmds::WebnnCommands>();
  if (c) {
    c->Init(commands_shm_id, commands_shm_offset, size);
  }
}

void InjectInstance(uint32_t instance_id, uint32_t instance_generation) {
  webnn::cmds::InjectInstance* c = GetCmdSpace<webnn::cmds::InjectInstance>();
  if (c) {
    c->Init(instance_id, instance_generation);
  }
}

void InjectDawnWireServer(int channel_id, int32_t route_id) {
  webnn::cmds::InjectDawnWireServer* c =
      GetCmdSpace<webnn::cmds::InjectDawnWireServer>();
  if (c) {
    c->Init(channel_id, route_id);
  }
}

#endif  // GPU_COMMAND_BUFFER_CLIENT_WEBNN_CMD_HELPER_AUTOGEN_H_

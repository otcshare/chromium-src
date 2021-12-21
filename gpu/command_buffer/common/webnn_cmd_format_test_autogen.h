// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file is auto-generated from
// gpu/command_buffer/build_webnn_cmd_buffer.py
// It's formatted by clang-format using chromium coding style:
//    clang-format -i -style=chromium filename
// DO NOT EDIT!

// This file contains unit tests for webnn commands
// It is included by webnn_cmd_format_test.cc

#ifndef GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_TEST_AUTOGEN_H_
#define GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_TEST_AUTOGEN_H_

TEST_F(WebNNFormatTest, WebnnCommands) {
  cmds::WebnnCommands& cmd = *GetBufferAs<cmds::WebnnCommands>();
  void* next_cmd =
      cmd.Set(&cmd, static_cast<uint32_t>(11), static_cast<uint32_t>(12),
              static_cast<uint32_t>(13));
  EXPECT_EQ(static_cast<uint32_t>(cmds::WebnnCommands::kCmdId),
            cmd.header.command);
  EXPECT_EQ(sizeof(cmd), cmd.header.size * 4u);
  EXPECT_EQ(static_cast<uint32_t>(11), cmd.commands_shm_id);
  EXPECT_EQ(static_cast<uint32_t>(12), cmd.commands_shm_offset);
  EXPECT_EQ(static_cast<uint32_t>(13), cmd.size);
  CheckBytesWrittenMatchesExpectedSize(next_cmd, sizeof(cmd));
}

TEST_F(WebNNFormatTest, InjectInstance) {
  cmds::InjectInstance& cmd = *GetBufferAs<cmds::InjectInstance>();
  void* next_cmd =
      cmd.Set(&cmd, static_cast<uint32_t>(11), static_cast<uint32_t>(12));
  EXPECT_EQ(static_cast<uint32_t>(cmds::InjectInstance::kCmdId),
            cmd.header.command);
  EXPECT_EQ(sizeof(cmd), cmd.header.size * 4u);
  EXPECT_EQ(static_cast<uint32_t>(11), cmd.instance_id);
  EXPECT_EQ(static_cast<uint32_t>(12), cmd.instance_generation);
  CheckBytesWrittenMatchesExpectedSize(next_cmd, sizeof(cmd));
}

TEST_F(WebNNFormatTest, InjectDawnWireServer) {
  cmds::InjectDawnWireServer& cmd = *GetBufferAs<cmds::InjectDawnWireServer>();
  void* next_cmd =
      cmd.Set(&cmd, static_cast<int>(11), static_cast<int32_t>(12));
  EXPECT_EQ(static_cast<uint32_t>(cmds::InjectDawnWireServer::kCmdId),
            cmd.header.command);
  EXPECT_EQ(sizeof(cmd), cmd.header.size * 4u);
  EXPECT_EQ(static_cast<int>(11), cmd.channel_id);
  EXPECT_EQ(static_cast<int32_t>(12), cmd.route_id);
  CheckBytesWrittenMatchesExpectedSize(next_cmd, sizeof(cmd));
}

#endif  // GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_TEST_AUTOGEN_H_

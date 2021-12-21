// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file is auto-generated from
// gpu/command_buffer/build_webnn_cmd_buffer.py
// It's formatted by clang-format using chromium coding style:
//    clang-format -i -style=chromium filename
// DO NOT EDIT!

#ifndef GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_AUTOGEN_H_
#define GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_AUTOGEN_H_

#define GL_SCANOUT_CHROMIUM 0x6000

struct WebnnCommands {
  typedef WebnnCommands ValueType;
  static const CommandId kCmdId = kWebnnCommands;
  static const cmd::ArgFlags kArgFlags = cmd::kFixed;
  static const uint8_t cmd_flags = CMD_FLAG_SET_TRACE_LEVEL(3);

  static uint32_t ComputeSize() {
    return static_cast<uint32_t>(sizeof(ValueType));  // NOLINT
  }

  void SetHeader() { header.SetCmd<ValueType>(); }

  void Init(uint32_t _commands_shm_id,
            uint32_t _commands_shm_offset,
            uint32_t _size) {
    SetHeader();
    commands_shm_id = _commands_shm_id;
    commands_shm_offset = _commands_shm_offset;
    size = _size;
  }

  void* Set(void* cmd,
            uint32_t _commands_shm_id,
            uint32_t _commands_shm_offset,
            uint32_t _size) {
    static_cast<ValueType*>(cmd)->Init(_commands_shm_id, _commands_shm_offset,
                                       _size);
    return NextCmdAddress<ValueType>(cmd);
  }

  gpu::CommandHeader header;
  uint32_t commands_shm_id;
  uint32_t commands_shm_offset;
  uint32_t size;
};

static_assert(sizeof(WebnnCommands) == 16,
              "size of WebnnCommands should be 16");
static_assert(offsetof(WebnnCommands, header) == 0,
              "offset of WebnnCommands header should be 0");
static_assert(offsetof(WebnnCommands, commands_shm_id) == 4,
              "offset of WebnnCommands commands_shm_id should be 4");
static_assert(offsetof(WebnnCommands, commands_shm_offset) == 8,
              "offset of WebnnCommands commands_shm_offset should be 8");
static_assert(offsetof(WebnnCommands, size) == 12,
              "offset of WebnnCommands size should be 12");

struct InjectInstance {
  typedef InjectInstance ValueType;
  static const CommandId kCmdId = kInjectInstance;
  static const cmd::ArgFlags kArgFlags = cmd::kFixed;
  static const uint8_t cmd_flags = CMD_FLAG_SET_TRACE_LEVEL(3);

  static uint32_t ComputeSize() {
    return static_cast<uint32_t>(sizeof(ValueType));  // NOLINT
  }

  void SetHeader() { header.SetCmd<ValueType>(); }

  void Init(uint32_t _instance_id, uint32_t _instance_generation) {
    SetHeader();
    instance_id = _instance_id;
    instance_generation = _instance_generation;
  }

  void* Set(void* cmd, uint32_t _instance_id, uint32_t _instance_generation) {
    static_cast<ValueType*>(cmd)->Init(_instance_id, _instance_generation);
    return NextCmdAddress<ValueType>(cmd);
  }

  gpu::CommandHeader header;
  uint32_t instance_id;
  uint32_t instance_generation;
};

static_assert(sizeof(InjectInstance) == 12,
              "size of InjectInstance should be 12");
static_assert(offsetof(InjectInstance, header) == 0,
              "offset of InjectInstance header should be 0");
static_assert(offsetof(InjectInstance, instance_id) == 4,
              "offset of InjectInstance instance_id should be 4");
static_assert(offsetof(InjectInstance, instance_generation) == 8,
              "offset of InjectInstance instance_generation should be 8");

struct InjectDawnWireServer {
  typedef InjectDawnWireServer ValueType;
  static const CommandId kCmdId = kInjectDawnWireServer;
  static const cmd::ArgFlags kArgFlags = cmd::kFixed;
  static const uint8_t cmd_flags = CMD_FLAG_SET_TRACE_LEVEL(3);

  static uint32_t ComputeSize() {
    return static_cast<uint32_t>(sizeof(ValueType));  // NOLINT
  }

  void SetHeader() { header.SetCmd<ValueType>(); }

  void Init(int _channel_id, int32_t _route_id) {
    SetHeader();
    channel_id = _channel_id;
    route_id = _route_id;
  }

  void* Set(void* cmd, int _channel_id, int32_t _route_id) {
    static_cast<ValueType*>(cmd)->Init(_channel_id, _route_id);
    return NextCmdAddress<ValueType>(cmd);
  }

  gpu::CommandHeader header;
  uint32_t channel_id;
  uint32_t route_id;
};

static_assert(sizeof(InjectDawnWireServer) == 12,
              "size of InjectDawnWireServer should be 12");
static_assert(offsetof(InjectDawnWireServer, header) == 0,
              "offset of InjectDawnWireServer header should be 0");
static_assert(offsetof(InjectDawnWireServer, channel_id) == 4,
              "offset of InjectDawnWireServer channel_id should be 4");
static_assert(offsetof(InjectDawnWireServer, route_id) == 8,
              "offset of InjectDawnWireServer route_id should be 8");

#endif  // GPU_COMMAND_BUFFER_COMMON_WEBNN_CMD_FORMAT_AUTOGEN_H_

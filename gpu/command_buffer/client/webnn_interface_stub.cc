// Copyright (c) 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "gpu/command_buffer/client/webnn_interface_stub.h"

namespace gpu {
namespace webnn {

namespace {

class APIChannelStub : public APIChannel {
 public:
  APIChannelStub() = default;

  const WebnnProcTable& GetProcs() const override { return procs_; }
  void Disconnect() override {}

  WebnnProcTable* procs() { return &procs_; }

 private:
  ~APIChannelStub() override = default;

  WebnnProcTable procs_ = {};
};

}  // anonymous namespace

WebNNInterfaceStub::WebNNInterfaceStub()
    : api_channel_(base::MakeRefCounted<APIChannelStub>()) {}

WebNNInterfaceStub::~WebNNInterfaceStub() = default;

WebnnProcTable* WebNNInterfaceStub::procs() {
  return static_cast<APIChannelStub*>(api_channel_.get())->procs();
}

// InterfaceBase implementation.
void WebNNInterfaceStub::GenSyncTokenCHROMIUM(GLbyte* sync_token) {}
void WebNNInterfaceStub::GenUnverifiedSyncTokenCHROMIUM(GLbyte* sync_token) {}
void WebNNInterfaceStub::VerifySyncTokensCHROMIUM(GLbyte** sync_tokens,
                                                   GLsizei count) {}
void WebNNInterfaceStub::WaitSyncTokenCHROMIUM(const GLbyte* sync_token) {}
void WebNNInterfaceStub::ShallowFlushCHROMIUM() {}

// WebNNInterface implementation
scoped_refptr<APIChannel> WebNNInterfaceStub::GetAPIChannel() const {
  return api_channel_;
}
void WebNNInterfaceStub::FlushCommands() {}
void WebNNInterfaceStub::EnsureAwaitingFlush(bool* needs_flush) {}
void WebNNInterfaceStub::FlushAwaitingCommands() {}
ReservedInstance WebNNInterfaceStub::ReserveInstance() {
  return {nullptr, 0, 0};
}
void WebNNInterfaceStub::InjectInstance(uint32_t id, uint32_t generation) {}
void WebNNInterfaceStub::InjectDawnWireServer(int channel_id, int32_t route_id) {}

// Include the auto-generated part of this class. We split this because
// it means we can easily edit the non-auto generated parts right here in
// this file instead of having to edit some template or the code generator.
#include "gpu/command_buffer/client/webnn_interface_stub_impl_autogen.h"

}  // namespace webnn
}  // namespace gpu

// Copyright (c) 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_CLIENT_WEBNN_INTERFACE_STUB_H_
#define GPU_COMMAND_BUFFER_CLIENT_WEBNN_INTERFACE_STUB_H_

#include "gpu/command_buffer/client/webnn_interface.h"

namespace gpu {
namespace webnn {

// This class a stub to help with mocks for the WebNNInterface class.
class WebNNInterfaceStub : public WebNNInterface {
 public:
  WebNNInterfaceStub();
  ~WebNNInterfaceStub() override;

  // InterfaceBase implementation.
  void GenSyncTokenCHROMIUM(GLbyte* sync_token) override;
  void GenUnverifiedSyncTokenCHROMIUM(GLbyte* sync_token) override;
  void VerifySyncTokensCHROMIUM(GLbyte** sync_tokens, GLsizei count) override;
  void WaitSyncTokenCHROMIUM(const GLbyte* sync_token) override;
  void ShallowFlushCHROMIUM() override;

  // WebNNInterface implementation
  scoped_refptr<APIChannel> GetAPIChannel() const override;
  void FlushCommands() override;
  void EnsureAwaitingFlush(bool* needs_flush) override;
  void FlushAwaitingCommands() override;
  ReservedInstance ReserveInstance() override;
  void InjectInstance(uint32_t id, uint32_t generation) override;
  void InjectDawnWireServer(int channel_id, int32_t route_id) override;

// Include the auto-generated part of this class. We split this because
// it means we can easily edit the non-auto generated parts right here in
// this file instead of having to edit some template or the code generator.
#include "gpu/command_buffer/client/webnn_interface_stub_autogen.h"

 protected:
  WebnnProcTable* procs();

 private:
  scoped_refptr<APIChannel> api_channel_;
};

}  // namespace webnn
}  // namespace gpu

#endif  // GPU_COMMAND_BUFFER_CLIENT_WEBNN_INTERFACE_STUB_H_

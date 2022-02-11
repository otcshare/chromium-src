// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_CLIENT_WEBNN_CLIENT_SERIALIZER_H_
#define GPU_COMMAND_BUFFER_CLIENT_WEBNN_CLIENT_SERIALIZER_H_

#include <webnn_wire/WireClient.h>

#include <memory>

#include "base/memory/raw_ptr.h"
#include "gpu/command_buffer/client/transfer_buffer.h"

namespace gpu {

class TransferBuffer;

namespace webnn {

class WebNNCmdHelper;
class WebNNImplementation;

class WebnnClientSerializer : public webnn_wire::CommandSerializer {
 public:
  WebnnClientSerializer(WebNNImplementation* client,
                       WebNNCmdHelper* helper,
                       std::unique_ptr<TransferBuffer> transfer_buffer);
  ~WebnnClientSerializer() override;

  // webnn_wire::CommandSerializer implementation
  size_t GetMaximumAllocationSize() const final;
  void* GetCmdSpace(size_t size) final;

  // Signal that it's important that the previously encoded commands are
  // flushed. Calling |AwaitingFlush| will return whether or not a flush still
  // needs to occur.
  void SetAwaitingFlush(bool awaiting_flush);

  // Check if the serializer has commands that have been serialized but not
  // flushed after |SetAwaitingFlush| was passed |true|.
  bool AwaitingFlush() const { return awaiting_flush_; }

  // Disconnect the serializer. Commands are forgotten and future calls to
  // |GetCmdSpace| will do nothing.
  void Disconnect();

  // Marks the commands' place in the GPU command buffer without flushing for
  // GPU execution.
  void Commit();

 private:
  // webnn_wire::CommandSerializer implementation
  bool Flush() final;

  raw_ptr<WebNNImplementation> client_;
  raw_ptr<WebNNCmdHelper> helper_;
  uint32_t put_offset_ = 0;
  std::unique_ptr<TransferBuffer> transfer_buffer_;
  uint32_t buffer_initial_size_;
  ScopedTransferBufferPtr buffer_;

  bool awaiting_flush_ = false;
};

}  // namespace webnn
}  // namespace gpu

#endif  // GPU_COMMAND_BUFFER_CLIENT_WEBNN_CLIENT_SERIALIZER_H_

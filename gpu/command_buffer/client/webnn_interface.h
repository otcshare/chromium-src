// Copyright (c) 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_CLIENT_WEBNN_INTERFACE_H_
#define GPU_COMMAND_BUFFER_CLIENT_WEBNN_INTERFACE_H_

#include <webnn/webnn_proc_table.h>
#include <webnn/webnn.h>

#include "base/callback.h"
#include "gpu/command_buffer/client/interface_base.h"
#include "gpu/command_buffer/common/webnn_cmd_enums.h"
#include "gpu/command_buffer/common/webnn_cmd_ids.h"

namespace gpu {
namespace webnn {

struct ReservedInstance {
  WNNInstance instance;
  uint32_t id;
  uint32_t generation;
};

// APIChannel is a RefCounted class which holds the Webnn wire client.
class APIChannel : public base::RefCounted<APIChannel> {
 public:
  // Get the proc table.
  // As long as a reference to this APIChannel alive, it is valid to
  // call these procs.
  virtual const WebnnProcTable& GetProcs() const = 0;

  // Disconnect. All commands using the WebNN API should become a
  // no-op and server-side resources can be freed.
  virtual void Disconnect() = 0;

 protected:
  friend class base::RefCounted<APIChannel>;
  APIChannel() = default;
  virtual ~APIChannel() = default;
};

class WebNNInterface : public InterfaceBase {
 public:
  WebNNInterface() = default;
  virtual ~WebNNInterface() = default;

  // Flush all commands.
  virtual void FlushCommands() = 0;

  // Ensure the awaiting flush flag is set on the client. Returns false
  // if a flush has already been indicated, or a flush is not needed (there may
  // be no commands to flush). Returns true if the caller should schedule a
  // flush.
  virtual void EnsureAwaitingFlush(bool* needs_flush) = 0;

  // If the awaiting flush flag is set, flushes commands. Otherwise, does
  // nothing.
  virtual void FlushAwaitingCommands() = 0;

  // Get a strong reference to the APIChannel backing the implementation.
  virtual scoped_refptr<APIChannel> GetAPIChannel() const = 0;

  virtual ReservedInstance ReserveInstance() = 0;
  virtual void InjectInstance(uint32_t id, uint32_t generation) = 0;
  virtual void InjectDawnWireServer(int channel_id, int32_t route_id) = 0;

// Include the auto-generated part of this class. We split this because
// it means we can easily edit the non-auto generated parts right here in
// this file instead of having to edit some template or the code generator.
#include "gpu/command_buffer/client/webnn_interface_autogen.h"
};

}  // namespace webnn
}  // namespace gpu

#endif  // GPU_COMMAND_BUFFER_CLIENT_WEBNN_INTERFACE_H_

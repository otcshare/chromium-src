// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "gpu/command_buffer/client/webnn_implementation.h"

#include <algorithm>
#include <vector>

#include <webnn/webnn_proc.h>

#include "base/numerics/checked_math.h"
#include "base/run_loop.h"
#include "base/trace_event/trace_event.h"
// #include "gpu/command_buffer/client/dawn_client_memory_transfer_service.h"
#include "gpu/command_buffer/client/gpu_control.h"
#include "gpu/command_buffer/client/shared_memory_limits.h"
#include "gpu/command_buffer/client/webnn_client_serializer.h"

#define GPU_CLIENT_SINGLE_THREAD_CHECK()

namespace gpu {
namespace webnn {

class WebnnWireServices : public APIChannel {
 private:
  friend class base::RefCounted<WebnnWireServices>;
  ~WebnnWireServices() override = default;

 public:
  WebnnWireServices(WebNNImplementation* webnn_implementation,
                    WebNNCmdHelper* helper,
                    std::unique_ptr<TransferBuffer> transfer_buffer)
      : serializer_(webnn_implementation, helper, std::move(transfer_buffer)),
        wire_client_(webnn_wire::WireClientDescriptor{
            &serializer_,
        }) {}

  const WebnnProcTable& GetProcs() const override {
    return webnn_wire::client::GetProcs();
  }

  webnn_wire::WireClient* wire_client() { return &wire_client_; }
  WebnnClientSerializer* serializer() { return &serializer_; }

  void Disconnect() override {
    disconnected_ = true;
    wire_client_.Disconnect();
    serializer_.Disconnect();
  }

  bool IsDisconnected() const { return disconnected_; }

 private:
  bool disconnected_ = false;
  WebnnClientSerializer serializer_;
  webnn_wire::WireClient wire_client_;
};

// Include the auto-generated part of this file. We split this because it means
// we can easily edit the non-auto generated parts right here in this file
// instead of having to edit some template or the code generator.
#include "gpu/command_buffer/client/webnn_implementation_impl_autogen.h"

WebNNImplementation::WebNNImplementation(
    WebNNCmdHelper* helper,
    TransferBufferInterface* transfer_buffer,
    GpuControl* gpu_control)
    : ImplementationBase(helper, transfer_buffer, gpu_control),
      helper_(helper) {}

WebNNImplementation::~WebNNImplementation() {
  LoseContext();

  // Wait for commands to finish before we continue destruction.
  // WebNNImplementation no longer owns the WebNN transfer buffer, but still
  // owns the GPU command buffer. We should not free shared memory that the
  // GPU process is using.
  helper_->Finish();
}

void WebNNImplementation::LoseContext() {
  lost_ = true;
  webnn_wire_->Disconnect();
}

gpu::ContextResult WebNNImplementation::Initialize(
    const SharedMemoryLimits& limits) {
  TRACE_EVENT0("gpu", "WebNNImplementation::Initialize");
  auto result = ImplementationBase::Initialize(limits);
  if (result != gpu::ContextResult::kSuccess) {
    return result;
  }

  std::unique_ptr<TransferBuffer> transfer_buffer =
      std::make_unique<TransferBuffer>(helper_);
  if (!transfer_buffer->Initialize(limits.start_transfer_buffer_size,
                                   /* start offset */ 0,
                                   limits.min_transfer_buffer_size,
                                   limits.max_transfer_buffer_size,
                                   /* alignment */ 8)) {
    return gpu::ContextResult::kFatalFailure;
  }

  webnn_wire_ = base::MakeRefCounted<WebnnWireServices>(
      this, helper_, std::move(transfer_buffer));

  // TODO(senorblanco): Do this only once per process. Doing it once per
  // WebNNImplementation is non-optimal but valid, since the returned
  // procs are always the same.
  webnnProcSetProcs(&webnn_wire::client::GetProcs());

  return gpu::ContextResult::kSuccess;
}

// ContextSupport implementation.
void WebNNImplementation::SetAggressivelyFreeResources(
    bool aggressively_free_resources) {
  NOTIMPLEMENTED();
}
void WebNNImplementation::Swap(uint32_t flags,
                               SwapCompletedCallback complete_callback,
                               PresentationCallback presentation_callback) {
  NOTIMPLEMENTED();
}
void WebNNImplementation::SwapWithBounds(
    const std::vector<gfx::Rect>& rects,
    uint32_t flags,
    SwapCompletedCallback swap_completed,
    PresentationCallback presentation_callback) {
  NOTIMPLEMENTED();
}
void WebNNImplementation::PartialSwapBuffers(
    const gfx::Rect& sub_buffer,
    uint32_t flags,
    SwapCompletedCallback swap_completed,
    PresentationCallback presentation_callback) {
  NOTIMPLEMENTED();
}
void WebNNImplementation::CommitOverlayPlanes(
    uint32_t flags,
    SwapCompletedCallback swap_completed,
    PresentationCallback presentation_callback) {
  NOTREACHED();
}
void WebNNImplementation::ScheduleOverlayPlane(
    int plane_z_order,
    gfx::OverlayTransform plane_transform,
    unsigned overlay_texture_id,
    const gfx::Rect& display_bounds,
    const gfx::RectF& uv_rect,
    bool enable_blend,
    unsigned gpu_fence_id) {
  NOTREACHED();
}
uint64_t WebNNImplementation::ShareGroupTracingGUID() const {
  NOTIMPLEMENTED();
  return 0;
}
void WebNNImplementation::SetErrorMessageCallback(
    base::RepeatingCallback<void(const char*, int32_t)> callback) {
  NOTIMPLEMENTED();
}
bool WebNNImplementation::ThreadSafeShallowLockDiscardableTexture(
    uint32_t texture_id) {
  NOTREACHED();
  return false;
}
void WebNNImplementation::CompleteLockDiscardableTexureOnContextThread(
    uint32_t texture_id) {
  NOTREACHED();
}
bool WebNNImplementation::ThreadsafeDiscardableTextureIsDeletedForTracing(
    uint32_t texture_id) {
  NOTREACHED();
  return false;
}
void* WebNNImplementation::MapTransferCacheEntry(uint32_t serialized_size) {
  NOTREACHED();
  return nullptr;
}
void WebNNImplementation::UnmapAndCreateTransferCacheEntry(uint32_t type,
                                                           uint32_t id) {
  NOTREACHED();
}
bool WebNNImplementation::ThreadsafeLockTransferCacheEntry(uint32_t type,
                                                           uint32_t id) {
  NOTREACHED();
  return false;
}
void WebNNImplementation::UnlockTransferCacheEntries(
    const std::vector<std::pair<uint32_t, uint32_t>>& entries) {
  NOTREACHED();
}
void WebNNImplementation::DeleteTransferCacheEntry(uint32_t type, uint32_t id) {
  NOTREACHED();
}
unsigned int WebNNImplementation::GetTransferBufferFreeSize() const {
  NOTREACHED();
  return 0;
}
bool WebNNImplementation::IsJpegDecodeAccelerationSupported() const {
  NOTREACHED();
  return false;
}
bool WebNNImplementation::IsWebPDecodeAccelerationSupported() const {
  NOTREACHED();
  return false;
}
bool WebNNImplementation::CanDecodeWithHardwareAcceleration(
    const cc::ImageHeaderMetadata* image_metadata) const {
  NOTREACHED();
  return false;
}

// InterfaceBase implementation.
void WebNNImplementation::GenSyncTokenCHROMIUM(GLbyte* sync_token) {
  // Need to commit the commands to the GPU command buffer first for SyncToken
  // to work.
  webnn_wire_->serializer()->Commit();
  ImplementationBase::GenSyncToken(sync_token);
}
void WebNNImplementation::GenUnverifiedSyncTokenCHROMIUM(GLbyte* sync_token) {
  // Need to commit the commands to the GPU command buffer first for SyncToken
  // to work.
  webnn_wire_->serializer()->Commit();
  ImplementationBase::GenUnverifiedSyncToken(sync_token);
}
void WebNNImplementation::VerifySyncTokensCHROMIUM(GLbyte** sync_tokens,
                                                   GLsizei count) {
  ImplementationBase::VerifySyncTokens(sync_tokens, count);
}
void WebNNImplementation::WaitSyncTokenCHROMIUM(const GLbyte* sync_token) {
  // Need to commit the commands to the GPU command buffer first for SyncToken
  // to work.
  webnn_wire_->serializer()->Commit();
  ImplementationBase::WaitSyncToken(sync_token);
}
void WebNNImplementation::ShallowFlushCHROMIUM() {
  FlushCommands();
}

bool WebNNImplementation::HasGrContextSupport() const {
  return true;
}

// ImplementationBase implementation.
void WebNNImplementation::IssueShallowFlush() {
  NOTIMPLEMENTED();
}

void WebNNImplementation::SetGLError(GLenum error,
                                     const char* function_name,
                                     const char* msg) {
  GPU_CLIENT_LOG("[" << GetLogPrefix() << "] Client Synthesized Error: "
                     << gles2::GLES2Util::GetStringError(error) << ": "
                     << function_name << ": " << msg);
  NOTIMPLEMENTED();
}

// GpuControlClient implementation.
void WebNNImplementation::OnGpuControlLostContext() {
  LoseContext();

  // This should never occur more than once.
  DCHECK(!lost_context_callback_run_);
  lost_context_callback_run_ = true;
  if (!lost_context_callback_.is_null()) {
    std::move(lost_context_callback_).Run();
  }
}
void WebNNImplementation::OnGpuControlLostContextMaybeReentrant() {
  // If this function is called, we are guaranteed to also get a call
  // to |OnGpuControlLostContext| when the callstack unwinds. Thus, this
  // function only handles immediately setting state so that other operations
  // which occur while the callstack is unwinding are aware that the context
  // is lost.
  lost_ = true;
}
void WebNNImplementation::OnGpuControlErrorMessage(const char* message,
                                                   int32_t id) {
  NOTIMPLEMENTED();
}
void WebNNImplementation::OnGpuControlSwapBuffersCompleted(
    const SwapBuffersCompleteParams& params,
    gfx::GpuFenceHandle release_fence) {
  NOTIMPLEMENTED();
}
void WebNNImplementation::OnSwapBufferPresented(
    uint64_t swap_id,
    const gfx::PresentationFeedback& feedback) {
  NOTIMPLEMENTED();
}
void WebNNImplementation::OnGpuControlReturnData(
    base::span<const uint8_t> data) {
  if (lost_) {
    return;
  }

  // static uint32_t return_trace_id = 0;
  // TRACE_EVENT_WITH_FLOW0(TRACE_DISABLED_BY_DEFAULT("gpu.webnn"),
  //                        "WebnnReturnCommands", return_trace_id++,
  //                        TRACE_EVENT_FLAG_FLOW_IN);

  // TRACE_EVENT1(TRACE_DISABLED_BY_DEFAULT("gpu.webnn"),
  //              "WebNNImplementation::OnGpuControlReturnData", "bytes",
  //              data.size());

  CHECK_GT(data.size(), sizeof(cmds::WebnnReturnDataHeader));

  const cmds::WebnnReturnDataHeader& webnnReturnDataHeader =
      *reinterpret_cast<const cmds::WebnnReturnDataHeader*>(data.data());

  switch (webnnReturnDataHeader.return_data_type) {
    case WebnnReturnDataType::kWebnnCommands: {
      CHECK_GE(data.size(), sizeof(cmds::WebnnReturnCommandsInfo));

      const cmds::WebnnReturnCommandsInfo* webnn_return_commands_info =
          reinterpret_cast<const cmds::WebnnReturnCommandsInfo*>(data.data());
      if (webnn_wire_->IsDisconnected()) {
        break;
      }
      // TODO(enga): Instead of a CHECK, this could generate a device lost
      // event on just that device. It doesn't seem worth doing right now
      // since a failure here is likely not recoverable.
      CHECK(webnn_wire_->wire_client()->HandleCommands(
          reinterpret_cast<const char*>(
              webnn_return_commands_info->deserialized_buffer),
          data.size() -
              offsetof(cmds::WebnnReturnCommandsInfo, deserialized_buffer)));
    } break;
    default:
      NOTREACHED();
  }
}

void WebNNImplementation::FlushCommands() {
  webnn_wire_->serializer()->Commit();
  helper_->Flush();
}

void WebNNImplementation::EnsureAwaitingFlush(bool* needs_flush) {
  // If there is already a flush waiting, we don't need to flush.
  // We only want to set |needs_flush| on state transition from
  // false -> true.
  if (webnn_wire_->serializer()->AwaitingFlush()) {
    *needs_flush = false;
    return;
  }

  // Set the state to waiting for flush, and then write |needs_flush|.
  // Could still be false if there's no data to flush.
  webnn_wire_->serializer()->SetAwaitingFlush(true);
  *needs_flush = webnn_wire_->serializer()->AwaitingFlush();
}

void WebNNImplementation::FlushAwaitingCommands() {
  if (webnn_wire_->serializer()->AwaitingFlush()) {
    webnn_wire_->serializer()->Commit();
    helper_->Flush();
  }
}

scoped_refptr<APIChannel> WebNNImplementation::GetAPIChannel() const {
  return webnn_wire_.get();
}

ReservedInstance WebNNImplementation::ReserveInstance() {
  // Commit because we need to make sure messages that free a previously used
  // texture are seen first. ReserveInstance may reuse an existing ID.
  webnn_wire_->serializer()->Commit();

  auto reservation = webnn_wire_->wire_client()->ReserveInstance();
  ReservedInstance result;
  result.instance = reservation.instance;
  result.id = reservation.id;
  result.generation = reservation.generation;
  return result;
}

void WebNNImplementation::InjectInstance(uint32_t id, uint32_t generation) {
  if (lost_) {
    return;
  }

  // Commit because we need to make sure messages that free a previously used
  // instance are seen first. ReserveInstance may reuse an existing ID.
  webnn_wire_->serializer()->Commit();

  helper_->InjectInstance(id, generation);
  helper_->Flush();
}

void WebNNImplementation::InjectDawnWireServer(int channel_id, int32_t route_id) {
  if (lost_) {
    return;
  }

  // Commit because we need to make sure messages that free a previously used
  // instance are seen first. ReserveInstance may reuse an existing ID.
  webnn_wire_->serializer()->Commit();

  helper_->InjectDawnWireServer(channel_id, route_id);
  helper_->Flush();
}

}  // namespace webnn
}  // namespace gpu

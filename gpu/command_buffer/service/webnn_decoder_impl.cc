// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "gpu/command_buffer/service/webnn_decoder_impl.h"

#include <webnn_native/WebnnNative.h>
#include <webnn_wire/WireServer.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "base/logging.h"
#include "base/memory/raw_ptr.h"
#include "base/numerics/checked_math.h"
#include "base/trace_event/trace_event.h"
#include "build/build_config.h"
#include "gpu/command_buffer/common/mailbox.h"
#include "gpu/command_buffer/common/webnn_cmd_format.h"
#include "gpu/command_buffer/service/command_buffer_service.h"
#include "gpu/command_buffer/service/decoder_client.h"
// #include "gpu/command_buffer/service/shared_image_factory.h"
#include "gpu/command_buffer/service/shared_image_manager.h"
#include "gpu/command_buffer/service/webnn_decoder.h"
#include "gpu/config/gpu_preferences.h"
#include "ipc/common/command_buffer_id.h"
#include "ipc/ipc_channel.h"

namespace gpu {
namespace webnn {

namespace {

constexpr size_t kMaxWireBufferSize =
    std::min(IPC::Channel::kMaximumMessageSize,
             static_cast<size_t>(1024 * 1024));

constexpr size_t kWebnnReturnCmdsOffset =
    offsetof(cmds::WebnnReturnCommandsInfo, deserialized_buffer);

static_assert(kWebnnReturnCmdsOffset < kMaxWireBufferSize, "");

class WireServerCommandSerializer : public webnn_wire::CommandSerializer {
 public:
  explicit WireServerCommandSerializer(DecoderClient* client);
  ~WireServerCommandSerializer() override = default;
  size_t GetMaximumAllocationSize() const final;
  void* GetCmdSpace(size_t size) final;
  bool Flush() final;

 private:
  raw_ptr<DecoderClient> client_;
  std::vector<uint8_t> buffer_;
  size_t put_offset_;
};

WireServerCommandSerializer::WireServerCommandSerializer(DecoderClient* client)
    : client_(client),
      buffer_(kMaxWireBufferSize),
      put_offset_(
          offsetof(cmds::WebnnReturnCommandsInfo, deserialized_buffer)) {
  // We prepopulate the message with the header and keep it between flushes so
  // we never need to write it again.
  cmds::WebnnReturnCommandsInfoHeader* header =
      reinterpret_cast<cmds::WebnnReturnCommandsInfoHeader*>(&buffer_[0]);
  header->return_data_header.return_data_type =
      WebnnReturnDataType::kWebnnCommands;
}

size_t WireServerCommandSerializer::GetMaximumAllocationSize() const {
  return kMaxWireBufferSize - kWebnnReturnCmdsOffset;
}

void* WireServerCommandSerializer::GetCmdSpace(size_t size) {
  // Note: Dawn will never call this function with |size| >
  // GetMaximumAllocationSize().
  DCHECK_LE(put_offset_, kMaxWireBufferSize);
  DCHECK_LE(size, GetMaximumAllocationSize());

  // Statically check that kMaxWireBufferSize + kMaxWireBufferSize is
  // a valid uint32_t. We can add put_offset_ and size without overflow.
  static_assert(base::CheckAdd(kMaxWireBufferSize, kMaxWireBufferSize)
                    .IsValid<uint32_t>(),
                "");
  uint32_t next_offset = put_offset_ + static_cast<uint32_t>(size);
  if (next_offset > buffer_.size()) {
    Flush();
    // TODO(enga): Keep track of how much command space the application is using
    // and adjust the buffer size accordingly.

    DCHECK_EQ(put_offset_, kWebnnReturnCmdsOffset);
    next_offset = put_offset_ + static_cast<uint32_t>(size);
  }

  uint8_t* ptr = &buffer_[put_offset_];
  put_offset_ = next_offset;
  return ptr;
}

bool WireServerCommandSerializer::Flush() {
  if (put_offset_ > kWebnnReturnCmdsOffset) {
    // TRACE_EVENT1(TRACE_DISABLED_BY_DEFAULT("gpu.webnn"),
    //              "WireServerCommandSerializer::Flush", "bytes", put_offset_);

    // static uint32_t return_trace_id = 0;
    // TRACE_EVENT_WITH_FLOW0(TRACE_DISABLED_BY_DEFAULT("gpu.webnn"),
    //                        "WebnnReturnCommands", return_trace_id++,
    //                        TRACE_EVENT_FLAG_FLOW_OUT);

    client_->HandleReturnData(base::make_span(buffer_.data(), put_offset_));
    put_offset_ = kWebnnReturnCmdsOffset;
  }
  return true;
}

}  // namespace

class WebNNDecoderImpl final : public WebNNDecoder {
 public:
  WebNNDecoderImpl(DecoderClient* client,
                   CommandBufferServiceBase* command_buffer_service,
                   SharedImageManager* shared_image_manager,
                   MemoryTracker* memory_tracker,
                   const GpuPreferences& gpu_preferences);

  WebNNDecoderImpl(const WebNNDecoderImpl&) = delete;
  WebNNDecoderImpl& operator=(const WebNNDecoderImpl&) = delete;

  ~WebNNDecoderImpl() override;

  // WebNNDecoder implementation
  ContextResult Initialize() override;

  // DecoderContext implementation.
  base::WeakPtr<DecoderContext> AsWeakPtr() override {
    return weak_ptr_factory_.GetWeakPtr();
  }
  const gles2::ContextState* GetContextState() override {
    NOTREACHED();
    return nullptr;
  }
  void Destroy(bool have_context) override;
  bool MakeCurrent() override { return true; }
  gl::GLContext* GetGLContext() override { return nullptr; }
  gl::GLSurface* GetGLSurface() override {
    NOTREACHED();
    return nullptr;
  }
  const gles2::FeatureInfo* GetFeatureInfo() const override {
    NOTREACHED();
    return nullptr;
  }
  Capabilities GetCapabilities() override { return {}; }
  void RestoreGlobalState() const override { NOTREACHED(); }
  void ClearAllAttributes() const override { NOTREACHED(); }
  void RestoreAllAttributes() const override { NOTREACHED(); }
  void RestoreState(const gles2::ContextState* prev_state) override {
    NOTREACHED();
  }
  void RestoreActiveTexture() const override { NOTREACHED(); }
  void RestoreAllTextureUnitAndSamplerBindings(
      const gles2::ContextState* prev_state) const override {
    NOTREACHED();
  }
  void RestoreActiveTextureUnitBinding(unsigned int target) const override {
    NOTREACHED();
  }
  void RestoreBufferBinding(unsigned int target) override { NOTREACHED(); }
  void RestoreBufferBindings() const override { NOTREACHED(); }
  void RestoreFramebufferBindings() const override { NOTREACHED(); }
  void RestoreRenderbufferBindings() override { NOTREACHED(); }
  void RestoreProgramBindings() const override { NOTREACHED(); }
  void RestoreTextureState(unsigned service_id) override { NOTREACHED(); }
  void RestoreTextureUnitBindings(unsigned unit) const override {
    NOTREACHED();
  }
  void RestoreVertexAttribArray(unsigned index) override { NOTREACHED(); }
  void RestoreAllExternalTextureBindingsIfNeeded() override { NOTREACHED(); }
  QueryManager* GetQueryManager() override {
    NOTREACHED();
    return nullptr;
  }
  void SetQueryCallback(unsigned int query_client_id,
                        base::OnceClosure callback) override {
    NOTREACHED();
  }
  gles2::GpuFenceManager* GetGpuFenceManager() override {
    NOTREACHED();
    return nullptr;
  }
  bool HasPendingQueries() const override { return false; }
  void ProcessPendingQueries(bool did_finish) override {}
  bool HasMoreIdleWork() const override { return false; }
  void PerformIdleWork() override {}

  bool HasPollingWork() const override { return false; }

  void PerformPollingWork() override { wire_serializer_->Flush(); }

  TextureBase* GetTextureBase(uint32_t client_id) override {
    NOTREACHED();
    return nullptr;
  }
  void SetLevelInfo(uint32_t client_id,
                    int level,
                    unsigned internal_format,
                    unsigned width,
                    unsigned height,
                    unsigned depth,
                    unsigned format,
                    unsigned type,
                    const gfx::Rect& cleared_rect) override {
    NOTREACHED();
  }
  bool WasContextLost() const override {
    NOTIMPLEMENTED();
    return false;
  }
  bool WasContextLostByRobustnessExtension() const override { return false; }
  void MarkContextLost(error::ContextLostReason reason) override {
    NOTIMPLEMENTED();
  }
  bool CheckResetStatus() override {
    NOTREACHED();
    return false;
  }
  void BeginDecoding() override {}
  void EndDecoding() override {}
  const char* GetCommandName(unsigned int command_id) const;
  error::Error DoCommands(unsigned int num_commands,
                          const volatile void* buffer,
                          int num_entries,
                          int* entries_processed) override;
  base::StringPiece GetLogPrefix() override { return "WebNNDecoderImpl"; }
  void BindImage(uint32_t client_texture_id,
                 uint32_t texture_target,
                 gl::GLImage* image,
                 bool can_bind_to_sampler) override {
    NOTREACHED();
  }
  gles2::ContextGroup* GetContextGroup() override { return nullptr; }
  gles2::ErrorState* GetErrorState() override {
    NOTREACHED();
    return nullptr;
  }
  std::unique_ptr<gles2::AbstractTexture> CreateAbstractTexture(
      GLenum target,
      GLenum internal_format,
      GLsizei width,
      GLsizei height,
      GLsizei depth,
      GLint border,
      GLenum format,
      GLenum type) override {
    NOTREACHED();
    return nullptr;
  }
  bool IsCompressedTextureFormat(unsigned format) override {
    NOTREACHED();
    return false;
  }
  bool ClearLevel(gles2::Texture* texture,
                  unsigned target,
                  int level,
                  unsigned format,
                  unsigned type,
                  int xoffset,
                  int yoffset,
                  int width,
                  int height) override {
    NOTREACHED();
    return false;
  }
  bool ClearCompressedTextureLevel(gles2::Texture* texture,
                                   unsigned target,
                                   int level,
                                   unsigned format,
                                   int width,
                                   int height) override {
    NOTREACHED();
    return false;
  }
  bool ClearCompressedTextureLevel3D(gles2::Texture* texture,
                                     unsigned target,
                                     int level,
                                     unsigned format,
                                     int width,
                                     int height,
                                     int depth) override {
    NOTREACHED();
    return false;
  }
  bool ClearLevel3D(gles2::Texture* texture,
                    unsigned target,
                    int level,
                    unsigned format,
                    unsigned type,
                    int width,
                    int height,
                    int depth) override {
    NOTREACHED();
    return false;
  }
  bool initialized() const override { return true; }
  void SetLogCommands(bool log_commands) override { NOTIMPLEMENTED(); }
  gles2::Outputter* outputter() const override {
    NOTIMPLEMENTED();
    return nullptr;
  }
  int GetRasterDecoderId() const override {
    NOTREACHED();
    return -1;
  }

 private:
  typedef error::Error (WebNNDecoderImpl::*CmdHandler)(
      uint32_t immediate_data_size,
      const volatile void* data);

  // A struct to hold info about each command.
  struct CommandInfo {
    CmdHandler cmd_handler;
    uint8_t arg_flags;   // How to handle the arguments for this command
    uint8_t cmd_flags;   // How to handle this command
    uint16_t arg_count;  // How many arguments are expected for this command.
  };

  // A table of CommandInfo for all the commands.
  static const CommandInfo command_info[kNumCommands - kFirstWebNNCommand];

// Generate a member function prototype for each command in an automated and
// typesafe way.
#define WEBNN_CMD_OP(name) \
  Error Handle##name(uint32_t immediate_data_size, const volatile void* data);
  WEBNN_COMMAND_LIST(WEBNN_CMD_OP)
#undef WEBNN_CMD_OP

  // The current decoder error communicates the decoder error through command
  // processing functions that do not return the error value. Should be set
  // only if not returning an error.
  error::Error current_decoder_error_ = error::kNoError;

  std::unique_ptr<webnn_native::Instance> webnn_instance_;

  std::unique_ptr<webnn_wire::WireServer> wire_server_;
  std::unique_ptr<WireServerCommandSerializer> wire_serializer_;

#if defined(WEBNN_ENABLE_GPU_BUFFER)
  SharedImageManager* shared_image_manager_;
#endif

  bool destroyed_ = false;

  base::WeakPtrFactory<WebNNDecoderImpl> weak_ptr_factory_{this};
};

constexpr WebNNDecoderImpl::CommandInfo WebNNDecoderImpl::command_info[] = {
#define WEBNN_CMD_OP(name)                                 \
  {                                                        \
      &WebNNDecoderImpl::Handle##name,                     \
      cmds::name::kArgFlags,                               \
      cmds::name::cmd_flags,                               \
      sizeof(cmds::name) / sizeof(CommandBufferEntry) - 1, \
  }, /* NOLINT */
    WEBNN_COMMAND_LIST(WEBNN_CMD_OP)
#undef WEBNN_CMD_OP
};

WebNNDecoder* CreateWebNNDecoderImpl(
    DecoderClient* client,
    CommandBufferServiceBase* command_buffer_service,
    SharedImageManager* shared_image_manager,
    MemoryTracker* memory_tracker,
    const GpuPreferences& gpu_preferences) {
  return new WebNNDecoderImpl(client, command_buffer_service,
                              shared_image_manager, memory_tracker,
                              gpu_preferences);
}

WebNNDecoderImpl::WebNNDecoderImpl(
    DecoderClient* client,
    CommandBufferServiceBase* command_buffer_service,
    SharedImageManager* shared_image_manager,
    MemoryTracker* memory_tracker,
    const GpuPreferences& gpu_preferences)
    : WebNNDecoder(client, command_buffer_service),
      webnn_instance_(new webnn_native::Instance()),
      wire_serializer_(new WireServerCommandSerializer(client))
#if defined(WEBNN_ENABLE_GPU_BUFFER)
      ,
      shared_image_manager_(shared_image_manager)
#endif
{
  webnn_wire::WireServerDescriptor descriptor = {};
  descriptor.procs = &webnn_native::GetProcs();
  descriptor.serializer = wire_serializer_.get();
  wire_server_ = std::make_unique<webnn_wire::WireServer>(descriptor);
}

WebNNDecoderImpl::~WebNNDecoderImpl() {
  Destroy(false);
}

void WebNNDecoderImpl::Destroy(bool have_context) {
  wire_server_ = nullptr;

  destroyed_ = true;
}

ContextResult WebNNDecoderImpl::Initialize() {
  return ContextResult::kSuccess;
}

const char* WebNNDecoderImpl::GetCommandName(unsigned int command_id) const {
  if (command_id >= kFirstWebNNCommand && command_id < kNumCommands) {
    return webnn::GetCommandName(static_cast<CommandId>(command_id));
  }
  return GetCommonCommandName(static_cast<cmd::CommandId>(command_id));
}

error::Error WebNNDecoderImpl::DoCommands(unsigned int num_commands,
                                          const volatile void* buffer,
                                          int num_entries,
                                          int* entries_processed) {
  DCHECK(entries_processed);
  int commands_to_process = num_commands;
  error::Error result = error::kNoError;
  const volatile CommandBufferEntry* cmd_data =
      static_cast<const volatile CommandBufferEntry*>(buffer);
  int process_pos = 0;
  CommandId command = static_cast<CommandId>(0);

  while (process_pos < num_entries && result == error::kNoError &&
         commands_to_process--) {
    const unsigned int size = cmd_data->value_header.size;
    command = static_cast<CommandId>(cmd_data->value_header.command);

    if (size == 0) {
      result = error::kInvalidSize;
      break;
    }

    if (static_cast<int>(size) + process_pos > num_entries) {
      result = error::kOutOfBounds;
      break;
    }

    const unsigned int arg_count = size - 1;
    unsigned int command_index = command - kFirstWebNNCommand;
    if (command_index < std::size(command_info)) {
      // Prevent all further WebNN commands from being processed if the server
      // is destroyed.
      if (destroyed_) {
        result = error::kLostContext;
        break;
      }
      const CommandInfo& info = command_info[command_index];
      unsigned int info_arg_count = static_cast<unsigned int>(info.arg_count);
      if ((info.arg_flags == cmd::kFixed && arg_count == info_arg_count) ||
          (info.arg_flags == cmd::kAtLeastN && arg_count >= info_arg_count)) {
        uint32_t immediate_data_size = (arg_count - info_arg_count) *
                                       sizeof(CommandBufferEntry);  // NOLINT
        result = (this->*info.cmd_handler)(immediate_data_size, cmd_data);
      } else {
        result = error::kInvalidArguments;
      }
    } else {
      result = DoCommonCommand(command, arg_count, cmd_data);
    }

    if (result == error::kNoError &&
        current_decoder_error_ != error::kNoError) {
      result = current_decoder_error_;
      current_decoder_error_ = error::kNoError;
    }

    if (result != error::kDeferCommandUntilLater) {
      process_pos += size;
      cmd_data += size;
    }
  }

  *entries_processed = process_pos;

  if (error::IsError(result)) {
    LOG(ERROR) << "Error: " << result << " for Command "
               << GetCommandName(command);
  }

  return result;
}

error::Error WebNNDecoderImpl::HandleInjectInstance(
    uint32_t immediate_data_size,
    const volatile void* cmd_data) {
  const volatile webnn::cmds::InjectInstance& c =
      *static_cast<const volatile webnn::cmds::InjectInstance*>(cmd_data);
  uint32_t id = static_cast<uint32_t>(c.instance_id);
  uint32_t generation = static_cast<uint32_t>(c.instance_generation);

  // Inject the instance in the webnn_wire::Server.
  if (!wire_server_->InjectInstance(webnn_instance_->Get(), id, generation)) {
    DLOG(ERROR) << "Failed to inject instance.";
    return error::kInvalidArguments;
  }

  return error::kNoError;
}

error::Error WebNNDecoderImpl::HandleInjectDawnWireServer(
    uint32_t immediate_data_size,
    const volatile void* cmd_data) {
#if defined(WEBNN_ENABLE_GPU_BUFFER)
  const volatile webnn::cmds::InjectDawnWireServer& c =
      *static_cast<const volatile webnn::cmds::InjectDawnWireServer*>(cmd_data);
  uint32_t channel_id = static_cast<uint32_t>(c.channel_id);
  uint32_t route_id = static_cast<uint32_t>(c.route_id);

  dawn_wire::WireServer* dawn_wire_server =
      shared_image_manager_->DawnWireServer(
          CommandBufferIdFromChannelAndRoute(channel_id, route_id));
  if (!wire_server_->InjectDawnWireServer(dawn_wire_server)) {
    DLOG(ERROR) << "Failed to inject instance.";
    return error::kInvalidArguments;
  }
#endif

  return error::kNoError;
}

error::Error WebNNDecoderImpl::HandleWebnnCommands(
    uint32_t immediate_data_size,
    const volatile void* cmd_data) {
  const volatile webnn::cmds::WebnnCommands& c =
      *static_cast<const volatile webnn::cmds::WebnnCommands*>(cmd_data);
  uint32_t size = static_cast<uint32_t>(c.size);
  uint32_t commands_shm_id = static_cast<uint32_t>(c.commands_shm_id);
  uint32_t commands_shm_offset = static_cast<uint32_t>(c.commands_shm_offset);

  const volatile char* shm_commands = GetSharedMemoryAs<const volatile char*>(
      commands_shm_id, commands_shm_offset, size);
  if (shm_commands == nullptr) {
    return error::kOutOfBounds;
  }

  // TRACE_EVENT_WITH_FLOW0(
  //     TRACE_DISABLED_BY_DEFAULT("gpu.webnn"), "WebnnCommands",
  //     (static_cast<uint64_t>(commands_shm_id) << 32) + commands_shm_offset,
  //     TRACE_EVENT_FLAG_FLOW_IN);

  // TRACE_EVENT1(TRACE_DISABLED_BY_DEFAULT("gpu.webnn"),
  //              "WebNNDecoderImpl::HandleWebnnCommands", "bytes", size);

  if (!wire_server_->HandleCommands(shm_commands, size)) {
    NOTREACHED();
    return error::kLostContext;
  }

  // TODO(crbug.com/1174145): This is O(N) where N is the number of devices.
  // Multiple submits would be O(N*M). We should find a way to more
  // intelligently poll for work on only the devices that need it.
  PerformPollingWork();

  return error::kNoError;
}

}  // namespace webnn
}  // namespace gpu

// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/ml/execution_impl_nn.h"

#include <memory>
#include <utility>

#include "base/logging.h"
#include "mojo/public/cpp/bindings/strong_binding.h"
#include "services/ml/ienn_symbol_table.h"
#include "services/ml/public/mojom/constants.mojom.h"

namespace ml {

ExecutionImplNN::ExecutionImplNN(const CompilationImplNN* compilation,
                                 mojo::ScopedSharedBufferHandle memory)
    : operands_(compilation->operands_),
      operations_(compilation->operations_),
      inputs_(compilation->inputs_),
      outputs_(compilation->outputs_),
      memory_(std::move(memory)),
      compilation_impl_(compilation) {
#if defined(OS_LINUX) || defined(OS_WIN)
  // Create Execution
  IE(ie_execution_create)(compilation_impl_->ie_compilation_, &ie_execution_);
#endif
  uint32_t total_length = 0;
  inputs_info_.reserve(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    uint32_t offset = total_length;
    uint32_t length = operands_[inputs_[i]].requiredSize();
    inputs_info_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
    total_length += length;
  }

  outputs_info_.reserve(outputs_.size());
  for (size_t i = 0; i < outputs_.size(); ++i) {
    uint32_t offset = total_length;
    uint32_t length = operands_[outputs_[i]].requiredSize();
    outputs_info_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
    total_length += length;
  }
}

ExecutionImplNN::~ExecutionImplNN() {
#if defined(OS_ANDROID)
#else
  IE(ie_execution_free)(ie_execution_);
#endif
  DLOG(INFO) << "ANeuralNetworksCompilation_free";
}

void ExecutionImplNN::ResetWithUserBuffer(mojom::UserBufferPtr user_buffer) {
  inputs_info_.clear();
  memory_ = std::move(user_buffer->memory);
  for (auto& input : user_buffer->inputs) {
    uint32_t offset = input->offset;
    uint32_t length = input->length;
    inputs_info_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
  }

  outputs_info_.clear();
  for (auto& output : user_buffer->outputs) {
    uint32_t offset = output->offset;
    uint32_t length = output->length;
    outputs_info_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
  }
}

void ExecutionImplNN::StartCompute(mojom::UserBufferPtr user_buffer,
                                   StartComputeCallback callback) {
  DLOG(INFO) << "ExecutionImplNN::StartCompute";
  if (user_buffer.get()) {
    ResetWithUserBuffer(std::move(user_buffer));
  }

  int32_t result = 0;
#if defined(OS_ANDROID)
  ANeuralNetworksExecution* nn_execution;
  result = ANeuralNetworksExecution_create(compilation_impl_->nn_compilation_,
                                           &nn_execution);
#endif
  for (size_t i = 0; i < inputs_info_.size(); ++i) {
    std::unique_ptr<OperandInfo>& info = inputs_info_[i];
#if defined(OS_ANDROID)
    result = ANeuralNetworksExecution_setInput(
        nn_execution, i, NULL, static_cast<void*>(info->mapping.get()),
        info->length);
#else
    result = IE(ie_execution_set_input)(ie_execution_, i, info->mapping.get(),
                                        info->length);
#endif
  }

  for (size_t i = 0; i < outputs_info_.size(); ++i) {
    std::unique_ptr<OperandInfo>& info = outputs_info_[i];
#if defined(OS_ANDROID)
    result = ANeuralNetworksExecution_setOutput(
        nn_execution, i, NULL, static_cast<void*>(info->mapping.get()),
        info->length);
#else
    result = IE(ie_execution_set_output)(ie_execution_, i, info->mapping.get(),
                                         info->length);
#endif
  }

#if defined(OS_ANDROID)
  ANeuralNetworksEvent* nn_event;
  result = ANeuralNetworksExecution_startCompute(nn_execution, &nn_event);
  LOG(INFO) << "ANeuralNetworksExecution_startCompute: " << result;
  result = ANeuralNetworksEvent_wait(nn_event);
  LOG(INFO) << "ANeuralNetworksEvent_wait: " << result;
  ANeuralNetworksEvent_free(nn_event);

  ANeuralNetworksExecution_free(nn_execution);
#else
  result = IE(ie_execution_start_compute)(ie_execution_);
#endif
  DLOG(INFO) << "ie_execution_start_compute: " << result;

  std::move(callback).Run(mojom::NOT_ERROR);
}
}  // namespace ml

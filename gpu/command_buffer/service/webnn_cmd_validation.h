// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Contains various validation functions for the webnn service.

#ifndef GPU_COMMAND_BUFFER_SERVICE_WEBNN_CMD_VALIDATION_H_
#define GPU_COMMAND_BUFFER_SERVICE_WEBNN_CMD_VALIDATION_H_

#include "gpu/command_buffer/common/webnn_cmd_enums.h"
#include "gpu/command_buffer/common/webnn_cmd_format.h"
#include "gpu/command_buffer/service/value_validator.h"

namespace gpu {
namespace webnn {

struct Validators {
  Validators();

#include "gpu/command_buffer/service/webnn_cmd_validation_autogen.h"
};

}  // namespace webnn
}  // namespace gpu

#endif  // GPU_COMMAND_BUFFER_SERVICE_WEBNN_CMD_VALIDATION_H_

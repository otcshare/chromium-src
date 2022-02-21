// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_OUT_OF_MEMORY_ERROR_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_OUT_OF_MEMORY_ERROR_H_

#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"

namespace blink {

class MLOutOfMemoryError : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static MLOutOfMemoryError* Create();
  MLOutOfMemoryError();

  MLOutOfMemoryError(const MLOutOfMemoryError&) = delete;
  MLOutOfMemoryError& operator=(const MLOutOfMemoryError&) = delete;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_OUT_OF_MEMORY_ERROR_H_

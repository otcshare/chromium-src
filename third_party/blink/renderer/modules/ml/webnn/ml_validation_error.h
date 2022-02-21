// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_VALIDATION_ERROR_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_VALIDATION_ERROR_H_

#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

class MLValidationError : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static MLValidationError* Create(const AtomicString& message);
  MLValidationError(const AtomicString& message);

  MLValidationError(const MLValidationError&) = delete;
  MLValidationError& operator=(const MLValidationError&) = delete;

  // gpu_validation_error.idl
  const String& message() const;

 private:
  String message_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_VALIDATION_ERROR_H_

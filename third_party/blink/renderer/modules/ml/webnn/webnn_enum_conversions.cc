// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_enum_conversions.h"

#include "base/check.h"
#include "base/notreached.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

template <>
WNNErrorFilter AsWebnnEnum<WNNErrorFilter>(const WTF::String& ml_enum) {
  if (ml_enum == "out-of-memory") {
    return WNNErrorFilter_OutOfMemory;
  }
  if (ml_enum == "validation") {
    return WNNErrorFilter_Validation;
  }
  NOTREACHED();
  return WNNErrorFilter_Force32;
}

}  // namespace blink

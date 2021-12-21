// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context.h"

#include "third_party/blink/renderer/modules/ml/ml.h"

namespace blink {

MLContext::MLContext(const AtomicString model_format, ML* ml)
    : model_format_(model_format), ml_(ml) {}

MLContext::~MLContext() = default;

AtomicString MLContext::modelFormat() const {
  return model_format_;
}

ML* MLContext::GetML() {
  return ml_.Get();
}

void MLContext::Trace(Visitor* visitor) const {
  visitor->Trace(ml_);

  ScriptWrappable::Trace(visitor);
}

}  // namespace blink

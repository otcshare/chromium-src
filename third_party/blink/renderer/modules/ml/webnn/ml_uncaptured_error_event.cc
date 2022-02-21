// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_uncaptured_error_event.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_uncaptured_error_event_init.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_mloutofmemoryerror_mlvalidationerror.h"

namespace blink {

// static
MLUncapturedErrorEvent* MLUncapturedErrorEvent::Create(
    const AtomicString& type,
    const MLUncapturedErrorEventInit* mlUncapturedErrorEventInitDict) {
  return MakeGarbageCollected<MLUncapturedErrorEvent>(
      type, mlUncapturedErrorEventInitDict);
}

MLUncapturedErrorEvent::MLUncapturedErrorEvent(
    const AtomicString& type,
    const MLUncapturedErrorEventInit* mlUncapturedErrorEventInitDict)
    : Event(type, Bubbles::kNo, Cancelable::kYes) {
  error_ = mlUncapturedErrorEventInitDict->error();
}

void MLUncapturedErrorEvent::Trace(Visitor* visitor) const {
  visitor->Trace(error_);
  Event::Trace(visitor);
}

const V8MLError* MLUncapturedErrorEvent::error() const {
  return error_;
}

}  // namespace blink

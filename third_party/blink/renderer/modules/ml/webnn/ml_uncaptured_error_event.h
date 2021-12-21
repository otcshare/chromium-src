// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_UNCAPTURED_ERROR_EVENT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_UNCAPTURED_ERROR_EVENT_H_

#include "third_party/blink/renderer/bindings/modules/v8/v8_typedefs.h"
#include "third_party/blink/renderer/modules/event_modules.h"

namespace blink {

class MLUncapturedErrorEventInit;

class MLUncapturedErrorEvent : public Event {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static MLUncapturedErrorEvent* Create(const AtomicString& type,
                                         const MLUncapturedErrorEventInit*);
  MLUncapturedErrorEvent(const AtomicString& type,
                          const MLUncapturedErrorEventInit*);

  MLUncapturedErrorEvent(const MLUncapturedErrorEvent&) = delete;
  MLUncapturedErrorEvent& operator=(const MLUncapturedErrorEvent&) = delete;

  void Trace(Visitor*) const override;

  // gpu_uncaptured_error_event.idl
  const V8MLError* error() const;

 private:
  Member<V8MLError> error_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_UNCAPTURED_ERROR_EVENT_H_

// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context.h"

#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"

namespace blink {

MLContext::MLContext(
    const V8MLDevicePreference device_preference,
    const V8MLPowerPreference power_preference,
    const V8MLModelFormat model_format,
    const unsigned int num_threads,
    ML* ml,
    ExecutionContext* execution_context,
    scoped_refptr<WebnnControlClientHolder> webnn_control_client,
    WNNContext webnn_context)
    : WebnnContext(execution_context, webnn_control_client, webnn_context),
      device_preference_(device_preference),
      power_preference_(power_preference),
      model_format_(model_format),
      num_threads_(num_threads),
      ml_(ml) {}

MLContext::MLContext(
    ML* ml,
    ExecutionContext* execution_context,
    scoped_refptr<WebnnControlClientHolder> webnn_control_client,
    WNNContext webnn_context)
    : WebnnContext(execution_context, webnn_control_client, webnn_context),
      device_preference_({}),
      power_preference_({}),
      model_format_({}),
      num_threads_(0),
      ml_(ml) {}

MLContext::~MLContext() = default;

V8MLDevicePreference MLContext::GetDevicePreference() const {
  return device_preference_;
}

V8MLPowerPreference MLContext::GetPowerPreference() const {
  return power_preference_;
}

V8MLModelFormat MLContext::GetModelFormat() const {
  return model_format_;
}

unsigned int MLContext::GetNumThreads() const {
  return num_threads_;
}

ML* MLContext::GetML() {
  return ml_.Get();
}

void MLContext::Trace(Visitor* visitor) const {
  visitor->Trace(ml_);

  ScriptWrappable::Trace(visitor);
  WebnnContext::Trace(visitor);
}

}  // namespace blink

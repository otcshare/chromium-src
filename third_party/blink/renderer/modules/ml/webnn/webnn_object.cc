// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"

#include "gpu/command_buffer/client/webnn_interface.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/platform/bindings/microtask.h"

namespace blink {

WebnnObjectBase::WebnnObjectBase(
    scoped_refptr<WebnnControlClientHolder> webnn_control_client)
    : webnn_control_client_(std::move(webnn_control_client)) {}

const scoped_refptr<WebnnControlClientHolder>&
WebnnObjectBase::GetWebnnControlClient() const {
  return webnn_control_client_;
}

void WebnnObjectBase::setLabel(const String& value) {
  // TODO: Relay label changes to Webnn
  label_ = value;
}

void WebnnObjectBase::EnsureFlush() {
  bool needs_flush = false;
  auto context_provider = GetContextProviderWeakPtr();
  if (UNLIKELY(!context_provider))
    return;
  context_provider->ContextProvider()->WebNNInterface()->EnsureAwaitingFlush(
      &needs_flush);
  if (!needs_flush) {
    // We've already enqueued a task to flush, or the command buffer
    // is empty. Do nothing.
    return;
  }
  Microtask::EnqueueMicrotask(WTF::Bind(
      [](scoped_refptr<WebnnControlClientHolder> webnn_control_client) {
        if (auto context_provider =
                webnn_control_client->GetContextProviderWeakPtr()) {
          context_provider->ContextProvider()
              ->WebNNInterface()
              ->FlushAwaitingCommands();
        }
      },
      webnn_control_client_));
}

// Flush commands up until now on this object's parent context immediately.
void WebnnObjectBase::FlushNow() {
  auto context_provider = GetContextProviderWeakPtr();
  if (LIKELY(context_provider)) {
    context_provider->ContextProvider()->WebNNInterface()->FlushCommands();
  }
}

WebnnObjectImpl::WebnnObjectImpl(MLContext* context)
    : WebnnObjectBase(context->GetWebnnControlClient()), context_(context) {}

WebnnObjectImpl::~WebnnObjectImpl() = default;

WNNContext WebnnObjectImpl::GetContextHandle() {
  return context_->GetHandle();
}

void WebnnObjectImpl::Trace(Visitor* visitor) const {
  visitor->Trace(context_);
  ScriptWrappable::Trace(visitor);
}

}  // namespace blink

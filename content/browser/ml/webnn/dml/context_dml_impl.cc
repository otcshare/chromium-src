// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/context_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/adapter_dml.h"
#include "content/browser/ml/webnn/dml/execution_context.h"
#include "content/browser/ml/webnn/dml/graph_dml_impl.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content::webnn {

ContextDMLImpl::~ContextDMLImpl() = default;

ContextDMLImpl::ContextDMLImpl(scoped_refptr<AdapterDML> adapter)
    : execution_context_(base::MakeRefCounted<ExecutionContext>(adapter)) {}

HRESULT ContextDMLImpl::Initialize() {
  HRESULT hr = execution_context_->Initialize();
  if (FAILED(hr)) {
    return hr;
  }
  return hr;
}

void ContextDMLImpl::CreateGraph(uint32_t graph_id,
                                 CreateGraphCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<ml::webnn::mojom::Graph> blink_remote;
  // The receiver bind to GraphDMLImpl.
  GraphDMLImpl::Create(blink_remote.InitWithNewPipeAndPassReceiver(),
                       execution_context_, graph_id);
  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace content::webnn

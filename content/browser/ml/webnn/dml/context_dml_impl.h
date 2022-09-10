// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_CONTEXT_DML_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_CONTEXT_DML_IMPL_H_

#include "components/ml/mojom/webnn_context.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/system/handle.h"

namespace content::webnn {

class AdapterDML;
class ExecutionContext;

class ContextDMLImpl : public ml::webnn::mojom::Context {
 public:
  ~ContextDMLImpl() override;
  explicit ContextDMLImpl(scoped_refptr<AdapterDML> adapter);

  ContextDMLImpl(const ContextDMLImpl&) = delete;
  ContextDMLImpl& operator=(const ContextDMLImpl&) = delete;

  HRESULT Initialize();

 private:
  // ml::webnn::mojom::context
  void CreateGraph(uint32_t, CreateGraphCallback) override;
  scoped_refptr<ExecutionContext> execution_context_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_CONTEXT_DML_IMPL_H_

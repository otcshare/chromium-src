// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_MOJO_SERVER_DML_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_MOJO_SERVER_DML_IMPL_H_

#include "base/memory/raw_ptr.h"
#include "components/ml/mojom/webnn_service.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"

namespace content::webnn {

class WebnnServiceDMLImpl;

class MojoServerDMLImpl : public ml::webnn::mojom::MojoServer {
 public:
  ~MojoServerDMLImpl() override;
  static void Create(
      mojo::PendingReceiver<ml::webnn::mojom::MojoServer> receiver,
      WebnnServiceDMLImpl* webnn_service);

  MojoServerDMLImpl(const MojoServerDMLImpl&) = delete;
  MojoServerDMLImpl& operator=(const MojoServerDMLImpl&) = delete;

 protected:
  explicit MojoServerDMLImpl(WebnnServiceDMLImpl* webnn_service);

 private:
  // ml::webnn::mojom::MojoServer
  void CreateContext(ml::webnn::mojom::ContextOptionsPtr options,
                     uint32_t context_id,
                     CreateContextCallback callback) override;
  // WebNN service is no destructor object that will not be outlived.
  raw_ptr<WebnnServiceDMLImpl> webnn_service_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_MOJO_SERVER_DML_IMPL_H_

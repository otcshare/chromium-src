// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_DML_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_DML_IMPL_H_

#include <map>

#include "components/ml/mojom/webnn_service.mojom.h"
#include "content/browser/ml/webnn/dml/adapter_dml.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/receiver.h"

namespace content::webnn {

namespace {

using ml::model_loader::mojom::PowerPreference;

}

class WebnnServiceDMLImpl : public ml::webnn::mojom::WebnnService {
 public:
  // The interface to create an `WebnnServiceDMLImpl` object and bind the mojo
  // receiver, called by the WebNN service factory.
  static void Create(
      mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver);

  explicit WebnnServiceDMLImpl(
      mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver);
  ~WebnnServiceDMLImpl() override;

  WebnnServiceDMLImpl(const WebnnServiceDMLImpl&) = delete;
  WebnnServiceDMLImpl& operator=(const WebnnServiceDMLImpl&) = delete;

  void BindMojoServer(
      mojo::PendingReceiver<ml::webnn::mojom::MojoServer> receiver) override;

  scoped_refptr<AdapterDML> RequestAdapter(PowerPreference power_preference);

 private:
  mojo::Receiver<ml::webnn::mojom::WebnnService> receiver_;
  std::map<AdapterType, scoped_refptr<AdapterDML>> adapter_map_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_DML_IMPL_H_

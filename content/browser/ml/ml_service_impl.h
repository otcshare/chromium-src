// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_ML_SERVICE_IMPL_H_
#define CONTENT_BROWSER_ML_ML_SERVICE_IMPL_H_

#include "content/common/content_export.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "third_party/blink/public/mojom/ml/ml.mojom.h"

namespace content {

class CONTENT_EXPORT MLServiceImpl : public ml::mojom::MLService {
 public:
  ~MLServiceImpl() override;
  // The interface to create an object, called by the ml service factory.
  static void Create(mojo::PendingReceiver<ml::mojom::MLService> receiver);
  MLServiceImpl(const MLServiceImpl&) = delete;
  MLServiceImpl& operator=(const MLServiceImpl&) = delete;

 protected:
  MLServiceImpl();

 private:
  // ml::mojom::MLService
  void Load(const std::vector<uint8_t>& model_content,
            ml::mojom::LoadModelOptionsPtr options,
            LoadCallback callback) override;
};

}  // namespace content
#endif  // CONTENT_BROWSER_ML_ML_SERVICE_IMPL_H_

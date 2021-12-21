// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_ML_MODEL_IMPL_H_
#define CONTENT_BROWSER_ML_ML_MODEL_IMPL_H_

#include <string>

#include "base/containers/flat_map.h"
#include "content/common/content_export.h"
#include "mojo/public/cpp/bindings/pending_remote.h"
#include "mojo/public/cpp/bindings/remote.h"
#include "third_party/blink/public/mojom/ml/ml.mojom.h"

namespace content {

class CONTENT_EXPORT MLModelImpl : public ml::mojom::ModelLoaded {
 public:
  // The interface to create an object.
  static void Create(const std::string& url,
                     ml::mojom::MLService::LoadCallback callback);

  MLModelImpl(const MLModelImpl&) = delete;
  MLModelImpl& operator=(const MLModelImpl&) = delete;

  ~MLModelImpl() override;

 protected:
  MLModelImpl();

 private:
  // ml::mojom::MLModel
  // This function does not return any prediction.
  void Compute(base::flat_map<std::string, ml::mojom::TensorPtr> input,
               ComputeCallback callback) override;
};

}  // namespace content

#endif  // CONTENT_BROWSER_ML_ML_MODEL_IMPL_H_

// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/ml_model_impl.h"

#include <memory>
#include <utility>
#include <vector>

#include "base/bind.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "third_party/abseil-cpp/absl/types/optional.h"

namespace content {

// static
void MLModelImpl::Create(const std::string& url,
                         ml::mojom::MLService::LoadCallback callback) {
  std::move(callback).Run(ml::mojom::LoadModelResult::kNotSupported,
                          mojo::NullRemote());
}

MLModelImpl::MLModelImpl() = default;

MLModelImpl::~MLModelImpl() = default;

void MLModelImpl::Compute(
    base::flat_map<std::string, ml::mojom::TensorPtr> input,
    ComputeCallback callback) {
  std::move(callback).Run(ml::mojom::ComputeResult::kNotSupported,
                          absl::nullopt);
}

}  // namespace content

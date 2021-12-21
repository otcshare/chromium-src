// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/ml_service_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "third_party/blink/public/mojom/ml/ml.mojom.h"

namespace content {

using ml::mojom::LoadModelOptionsPtr;

// static
void MLServiceImpl::Create(
    mojo::PendingReceiver<ml::mojom::MLService> receiver) {
  mojo::MakeSelfOwnedReceiver<ml::mojom::MLService>(
      base::WrapUnique(new MLServiceImpl()), std::move(receiver));
}

MLServiceImpl::~MLServiceImpl() = default;

MLServiceImpl::MLServiceImpl() = default;

void MLServiceImpl::Load(const std::vector<uint8_t>& model_content,
                         LoadModelOptionsPtr options,
                         LoadCallback callback) {
  std::move(callback).Run(ml::mojom::LoadModelResult::kNotSupported,
                          mojo::NullRemote());
}

}  // namespace content

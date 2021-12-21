// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/ml_service_impl_cros.h"

#include <fstream>

#include "base/memory/ptr_util.h"
#include "content/browser/ml/ml_model_impl_cros.h"
#include "content/public/browser/render_frame_host.h"
#include "content/public/browser/storage_partition.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "third_party/blink/public/mojom/ml/ml.mojom.h"

namespace content {

// static
void CrOSMLServiceImpl::Create(
    mojo::PendingReceiver<ml::mojom::MLService> receiver) {
  mojo::MakeSelfOwnedReceiver<ml::mojom::MLService>(
      base::WrapUnique(new CrOSMLServiceImpl()), std::move(receiver));
}

CrOSMLServiceImpl::~CrOSMLServiceImpl() = default;

CrOSMLServiceImpl::CrOSMLServiceImpl() = default;

void CrOSMLServiceImpl::Load(const std::vector<uint8_t>& model_content,
                             ml::mojom::LoadModelOptionsPtr options,
                             LoadCallback callback) {
  base::flat_map<std::string, int> input_node_name_to_index,
      output_node_name_to_index;

  if (options.is_null() || options->inputs.is_null() ||
      options->outputs.is_null()) {
    std::move(callback).Run(ml::mojom::LoadModelResult::kUnknownError,
                            mojo::NullRemote());
    return;
  }

  // Currently, we only supports `names_to_indices`.
  if (!options->inputs->is_names_to_indices() ||
      !options->outputs->is_names_to_indices()) {
    std::move(callback).Run(ml::mojom::LoadModelResult::kNotSupported,
                            mojo::NullRemote());
    return;
  }

  for (const auto& name_indice : options->inputs->get_names_to_indices()) {
    input_node_name_to_index[name_indice.first] = name_indice.second;
  }
  for (const auto& name_indice : options->outputs->get_names_to_indices()) {
    output_node_name_to_index[name_indice.first] = name_indice.second;
  }

  CrOSMLModelImpl::Create(model_content, std::move(input_node_name_to_index),
                          std::move(output_node_name_to_index),
                          std::move(callback));
}

}  // namespace content

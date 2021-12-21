// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_ML_MODEL_IMPL_CROS_H_
#define CONTENT_BROWSER_ML_ML_MODEL_IMPL_CROS_H_

#include <string>

#include "base/containers/flat_map.h"
#include "chromeos/services/machine_learning/public/cpp/service_connection.h"
#include "chromeos/services/machine_learning/public/mojom/graph_executor.mojom.h"
#include "chromeos/services/machine_learning/public/mojom/machine_learning_service.mojom.h"
#include "chromeos/services/machine_learning/public/mojom/model.mojom.h"
#include "content/common/content_export.h"
#include "mojo/public/cpp/bindings/pending_remote.h"
#include "mojo/public/cpp/bindings/remote.h"
#include "third_party/blink/public/mojom/ml/ml.mojom.h"

namespace content {

class CrOSMLModelImpl final : public ml::mojom::ModelLoaded {
 public:
  // The interface to create an object.
  static void Create(const std::vector<uint8_t>& model_body,
                     base::flat_map<std::string, int> input_node_name_to_index,
                     base::flat_map<std::string, int> output_node_name_to_index,
                     ml::mojom::MLService::LoadCallback callback);
  CrOSMLModelImpl(const CrOSMLModelImpl&) = delete;
  CrOSMLModelImpl& operator=(const CrOSMLModelImpl&) = delete;
  CrOSMLModelImpl(
      mojo::Remote<chromeos::machine_learning::mojom::Model> model_remote_cros,
      mojo::Remote<chromeos::machine_learning::mojom::GraphExecutor>
          executor_remote_cros,
      std::vector<std::string> output_names);
  ~CrOSMLModelImpl() override;

 private:
  // Remotes used to execute functions in the ML service side.
  mojo::Remote<chromeos::machine_learning::mojom::Model> model_;
  mojo::Remote<chromeos::machine_learning::mojom::GraphExecutor> executor_;
  // Needed in calling CrOS's prediction API.
  std::vector<std::string> output_names_;
  // ml::mojom::ModelLoaded
  void Compute(base::flat_map<std::string, ml::mojom::TensorPtr> input,
               ComputeCallback callback) override;
  void OnComputeResult(
      ComputeCallback callback_blink,
      chromeos::machine_learning::mojom::ExecuteResult result_cros,
      absl::optional<std::vector<chromeos::machine_learning::mojom::TensorPtr>>
          predictions_cros);
};

}  // namespace content

#endif  // CONTENT_BROWSER_ML_ML_MODEL_IMPL_CROS_H_

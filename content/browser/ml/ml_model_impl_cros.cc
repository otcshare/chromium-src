// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include "content/browser/ml/ml_model_impl_cros.h"

#include <utility>
#include <vector>

#include "base/bind.h"
#include "chromeos/services/machine_learning/public/mojom/machine_learning_service.mojom.h"
#include "chromeos/services/machine_learning/public/mojom/tensor.mojom.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/network/public/cpp/resource_request.h"
#include "third_party/abseil-cpp/absl/types/optional.h"

namespace content {

namespace {

void OnExecutorCreated(
    mojo::Remote<chromeos::machine_learning::mojom::Model> model_remote_cros,
    mojo::Remote<chromeos::machine_learning::mojom::GraphExecutor>
        executor_remote_cros,
    std::vector<std::string> output_names,
    ml::mojom::MLService::LoadCallback callback,
    chromeos::machine_learning::mojom::CreateGraphExecutorResult result) {
  if (result !=
      chromeos::machine_learning::mojom::CreateGraphExecutorResult::OK) {
    std::move(callback).Run(ml::mojom::LoadModelResult::kUnknownError,
                            mojo::NullRemote());
    return;
  }
  mojo::PendingRemote<ml::mojom::ModelLoaded> renderer_remote;
  auto impl = base::WrapUnique(new CrOSMLModelImpl(
      std::move(model_remote_cros), std::move(executor_remote_cros),
      std::move(output_names)));
  mojo::MakeSelfOwnedReceiver<ml::mojom::ModelLoaded>(
      std::move(impl), renderer_remote.InitWithNewPipeAndPassReceiver());
  std::move(callback).Run(ml::mojom::LoadModelResult::kOk,
                          std::move(renderer_remote));
}

void OnModelLoaded(
    mojo::Remote<chromeos::machine_learning::mojom::Model> model_remote_cros,
    std::vector<std::string> output_names,
    ml::mojom::MLService::LoadCallback callback,
    chromeos::machine_learning::mojom::LoadModelResult result) {
  if (result != chromeos::machine_learning::mojom::LoadModelResult::OK) {
    auto error_code = ml::mojom::LoadModelResult::kUnknownError;
    if (result ==
        chromeos::machine_learning::mojom::LoadModelResult::MODEL_SPEC_ERROR) {
      error_code = ml::mojom::LoadModelResult::kWrongModelSpec;
    } else if (result == chromeos::machine_learning::mojom::LoadModelResult::
                             LOAD_MODEL_ERROR) {
      error_code = ml::mojom::LoadModelResult::kLoadModelError;
    }
    std::move(callback).Run(error_code, mojo::NullRemote());
    return;
  }
  // Then tries to create the graph executor.
  mojo::Remote<chromeos::machine_learning::mojom::GraphExecutor>
      executor_remote_cros;
  auto receiver = executor_remote_cros.BindNewPipeAndPassReceiver();
  auto options = chromeos::machine_learning::mojom::GraphExecutorOptions::New();
  options->use_nnapi = false;
  model_remote_cros->CreateGraphExecutorWithOptions(
      std::move(options), std::move(receiver),
      base::BindOnce(&OnExecutorCreated, std::move(model_remote_cros),
                     std::move(executor_remote_cros), std::move(output_names),
                     std::move(callback)));
}

}  // namespace


// static
void CrOSMLModelImpl::Create(
    const std::vector<uint8_t>& model_body,
    base::flat_map<std::string, int> input_node_name_to_index,
    base::flat_map<std::string, int> output_node_name_to_index,
    ml::mojom::MLService::LoadCallback callback) {
  mojo::Remote<chromeos::machine_learning::mojom::Model> model_remote_cros;
  // Does this first before moving output_node_name_to_index.
  std::vector<std::string> output_names;
  for (const auto& name_index : output_node_name_to_index)
  {
    output_names.push_back(name_index.first);
  }

  // Loads the model in ml-service.
  std::string model_string;
  model_string.resize(model_body.size());
  for (size_t i = 0; i < model_string.size(); i++) {
    model_string[i] = model_body[i];
  }
  auto spec = chromeos::machine_learning::mojom::FlatBufferModelSpec::New(
      std::move(model_string), std::move(input_node_name_to_index),
      std::move(output_node_name_to_index), "WebMLModel");
  auto model_receiver = model_remote_cros.BindNewPipeAndPassReceiver();
  chromeos::machine_learning::ServiceConnection::GetInstance()
      ->GetMachineLearningService()
      .LoadFlatBufferModel(
          std::move(spec), std::move(model_receiver),
          base::BindOnce(&OnModelLoaded, std::move(model_remote_cros),
                         std::move(output_names), std::move(callback)));
}

CrOSMLModelImpl::CrOSMLModelImpl(
    mojo::Remote<chromeos::machine_learning::mojom::Model> model_remote_cros,
    mojo::Remote<chromeos::machine_learning::mojom::GraphExecutor>
        executor_remote_cros,
    std::vector<std::string> output_names)
    : model_(std::move(model_remote_cros)),
      executor_(std::move(executor_remote_cros)),
      output_names_(std::move(output_names)) {}

CrOSMLModelImpl::~CrOSMLModelImpl() = default;

void CrOSMLModelImpl::Compute(
    base::flat_map<std::string, ml::mojom::TensorPtr> input_blink,
    ComputeCallback callback) {
  base::flat_map<std::string, chromeos::machine_learning::mojom::TensorPtr>
      input_cros;
  for (const auto& name_tensor : input_blink) {
    auto tensor_cros = chromeos::machine_learning::mojom::Tensor::New();
    // Convert the shape.
    auto shape_value = chromeos::machine_learning::mojom::Int64List::New();
    for (const auto num : name_tensor.second->dimensions->value) {
      shape_value->value.push_back(num);
    }
    tensor_cros->shape = std::move(shape_value);
    // Convert the value list.
    auto value_list_cros = chromeos::machine_learning::mojom::ValueList::New();
    if (name_tensor.second->data->is_float32_list()) {
      auto float_list = chromeos::machine_learning::mojom::FloatList::New();
      for (const auto num :
           name_tensor.second->data->get_float32_list()->value) {
        float_list->value.push_back(num);
      }
      value_list_cros->set_float_list(std::move(float_list));
    } else if (name_tensor.second->data->is_int64_list()) {
      auto int64_list = chromeos::machine_learning::mojom::Int64List::New();
      for (const auto num : name_tensor.second->data->get_int64_list()->value) {
        int64_list->value.push_back(num);
      }
      value_list_cros->set_int64_list(std::move(int64_list));
    } else {
      NOTREACHED();  // not implemented yet.
    }
    tensor_cros->data = std::move(value_list_cros);
    input_cros[name_tensor.first] = std::move(tensor_cros);
  }

  executor_->Execute(
      std::move(input_cros), output_names_,
      base::BindOnce(&CrOSMLModelImpl::OnComputeResult, base::Unretained(this),
                     std::move(callback)));
}

void CrOSMLModelImpl::OnComputeResult(
    ComputeCallback callback_blink,
    chromeos::machine_learning::mojom::ExecuteResult result_cros,
    absl::optional<std::vector<chromeos::machine_learning::mojom::TensorPtr>>
        predictions_cros) {
  if (result_cros != chromeos::machine_learning::mojom::ExecuteResult::OK ||
      !predictions_cros.has_value() ||
      predictions_cros.value().size() != output_names_.size()) {
    std::move(callback_blink)
        .Run(ml::mojom::ComputeResult::kUnknownError, absl::nullopt);
    return;
  }
  base::flat_map<std::string, ml::mojom::TensorPtr> name_to_prediction;
  for (size_t i = 0; i < predictions_cros.value().size(); ++i) {
    const auto& tensor_cros = predictions_cros.value()[i];
    auto tensor_blink = ml::mojom::Tensor::New();
    // Shape
    auto shape_blink = ml::mojom::Uint32List::New();
    // TODO (honglinyu): may lose precision here.
    for (const auto v : tensor_cros->shape->value) {
      shape_blink->value.push_back(v);
    }
    tensor_blink->dimensions = std::move(shape_blink);
    // Data
    auto data_blink = ml::mojom::ValueList::New();
    if (tensor_cros->data->is_float_list()) {
      auto float_list = ml::mojom::Float64List::New();
      float_list->value = tensor_cros->data->get_float_list()->value;
      data_blink->set_float64_list(std::move(float_list));
    } else if (tensor_cros->data->is_int64_list()) {
      auto int64_list = ml::mojom::Int64List::New();
      int64_list->value = tensor_cros->data->get_int64_list()->value;
      data_blink->set_int64_list(std::move(int64_list));
    } else {
      NOTREACHED();  // not implemented yet.
    }
    tensor_blink->data = std::move(data_blink);
    name_to_prediction[output_names_[i]] = std::move(tensor_blink);
  }

  std::move(callback_blink)
      .Run(ml::mojom::ComputeResult::kOk, std::move(name_to_prediction));
}

}  // namespace content

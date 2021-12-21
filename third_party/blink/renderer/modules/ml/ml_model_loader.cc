// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_model_loader.h"

#include "third_party/blink/renderer/modules/ml/ml_type_converters.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_load_model_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/ml_model_loaded.h"

namespace blink {

namespace {

void OnModelLoaded(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    ml::mojom::blink::LoadModelResult result,
    mojo::PendingRemote<ml::mojom::blink::ModelLoaded> pending_remote) {
  switch (result) {
    case ml::mojom::blink::LoadModelResult::kUnknownError: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Internal error."));
      return;
    }
    case ml::mojom::blink::LoadModelResult::kNotSupported: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kNotSupportedError, "Not supported."));
      return;
    }
    case ml::mojom::blink::LoadModelResult::kLoadModelError: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kNotSupportedError, "Can't load the model."));
      return;
    }
    case ml::mojom::blink::LoadModelResult::kWrongModelSpec: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kNotSupportedError, "Wrong model specification."));
      return;
    }
    case ml::mojom::blink::LoadModelResult::kOk: {
      auto* ml_model = MakeGarbageCollected<MLModelLoaded>(
          ExecutionContext::From(script_state), std::move(pending_remote));
      resolver->Resolve(ml_model);
      return;
    }
  }

  NOTREACHED() << "ML model loading returns an invalid result.";
}

} // namespace

MLModelLoader::MLModelLoader(ExecutionContext* execution_context,
                             MLContext* ml_context)
    : ml_context_(ml_context) {
  if (ml_context != nullptr) {
    model_format_ = ml_context->modelFormat();
  }
}

// static
MLModelLoader* MLModelLoader::Create(ScriptState* script_state,
                                     MLContext* ml_context,
                                     ExceptionState& exception_state) {
  if (ml_context == nullptr) {
    return nullptr;
  }

  ExecutionContext* execution_context = ExecutionContext::From(script_state);
  return MakeGarbageCollected<MLModelLoader>(execution_context, ml_context);
}

MLModelLoader::~MLModelLoader() = default;

ScriptPromise MLModelLoader::load(ScriptState* script_state,
                                  DOMArrayBuffer* buffer,
                                  const MLLoadModelOptions* options,
                                  ExceptionState& exception_state) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();

  if (ml_context_->GetML() == nullptr) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError, "Internal error."));
  } else if (buffer == nullptr || options == nullptr) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kConstraintError, "Invalid input arguments."));
  } else {
    // Transforms model content.
    Vector<uint8_t> model_content;
    model_content.resize(
        static_cast<wtf_size_t>(buffer->ByteLength()));
    for (wtf_size_t i = 0; i < model_content.size(); i++) {
      model_content[i] = static_cast<uint8_t*>(buffer->Data())[i];
    }
    // Transforms load model options.
    auto options_mojo =
        mojo::ConvertTo<ml::mojom::blink::LoadModelOptionsPtr>(options);

    ml_context_->GetML()->Load(
        script_state, model_content, std::move(options_mojo), exception_state,
        WTF::Bind(&OnModelLoaded, WrapPersistent(script_state),
                  WrapPersistent(resolver)));
  }

  return promise;
}

void MLModelLoader::Trace(Visitor* visitor) const {
  visitor->Trace(ml_context_);

  ScriptWrappable::Trace(visitor);
}

}  // namespace blink

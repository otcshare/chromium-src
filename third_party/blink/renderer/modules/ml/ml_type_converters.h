// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_TYPE_CONVERTERS_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_TYPE_CONVERTERS_H_

#include "mojo/public/cpp/bindings/type_converter.h"
#include "third_party/blink/public/mojom/ml/ml.mojom-blink-forward.h"
#include "third_party/blink/renderer/modules/modules_export.h"

namespace blink {

class MLLoadModelOptions;
class MLTensor;

}  // namespace blink

namespace mojo {

// Converters from IDL to Mojo.
template <>
struct MODULES_EXPORT TypeConverter<ml::mojom::blink::LoadModelOptionsPtr,
                                    blink::MLLoadModelOptions*> {
  static ml::mojom::blink::LoadModelOptionsPtr Convert(
      const blink::MLLoadModelOptions* input);
};

template <>
struct MODULES_EXPORT
    TypeConverter<ml::mojom::blink::TensorPtr, blink::MLTensor*> {
  static ml::mojom::blink::TensorPtr Convert(const blink::MLTensor* input);
};

// Converters from Mojo to IDL.
template <>
struct MODULES_EXPORT
    TypeConverter<blink::MLTensor*, ml::mojom::blink::TensorPtr> {
  static blink::MLTensor* Convert(const ml::mojom::blink::TensorPtr& input);
};

}  // namespace mojo

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_TYPE_CONVERTERS_H_

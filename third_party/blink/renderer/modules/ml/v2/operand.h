// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPERAND_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPERAND_H_

#include "base/logging.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"

namespace blink {

class NNModel;
class Operand : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  Operand() = default;
  ~Operand() override = default;

  // First/NextInput are used for getting inputs when traversaling model tree.
  virtual Operand* FirstInput() const;
  virtual Operand* NextInput();

  // Add operand into Mojo::ModelInfo, operand value to buffer_views that will
  // be copied to Mojo::ModelInfo->memory which is total memory, and map
  // input/output name to index that will be used in execution phase.
  virtual void AddLayer(NNModel* model, uint32_t& index);
  virtual Vector<uint32_t> GetDimensions();

  void SetIndex(uint32_t index);
  uint32_t Index();
  void SetTraversal(bool traversal);
  bool Traversal();

  // Interface required by garbage collection.
  void Trace(Visitor*) const override;

 private:
  // index of operand in the model.
  uint32_t index_;
  // The traversalled operand doesn't traversal again.
  bool traversalled_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPERAND_H_

// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBNN_WEBNN_OBJECT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBNN_WEBNN_OBJECT_H_

#include <webnn/webnn.h>
#include <webnn/webnn_proc_table.h>

#include "base/memory/scoped_refptr.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/graphics/gpu/webnn_control_client_holder.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

#define WEBNN_OBJECTS               \
  X(Context, context)               \
  X(GraphBuilder, graphBuilder)     \
  X(Graph, graph)                   \
  X(Instance, instance)             \
  X(NamedInputs, namedInputs)       \
  X(NamedOperands, namedOperands)   \
  X(NamedOutputs, namedOutputs)     \
  X(Operand, operand)               \
  X(FusionOperator, fusionOperator) \
  X(OperandArray, operandArray)     \
  X(OperatorArray, operatorArray)

namespace gpu {
namespace webnn {

class WebNNInterface;

}  // namespace webnn
}  // namespace gpu

namespace blink {

template <typename T>
struct WNNReleaseFn;

#define X(Name, name)                                         \
  template <>                                                 \
  struct WNNReleaseFn<WNN##Name> {                            \
    static constexpr void (*WebnnProcTable::*fn)(WNN##Name) = \
        &WebnnProcTable::name##Release;                       \
  };
WEBNN_OBJECTS
#undef X

class MLContext;

// This class allows objects to hold onto a WebnnControlClientHolder.
// The WebnnControlClientHolder is used to hold the WebNNInterface and keep
// track of whether or not the client has been destroyed. If the client is
// destroyed, we should not call any Webnn functions.
class WebnnObjectBase {
 public:
  explicit WebnnObjectBase(
      scoped_refptr<WebnnControlClientHolder> webnn_control_client);

  const scoped_refptr<WebnnControlClientHolder>& GetWebnnControlClient() const;
  base::WeakPtr<WebGraphicsContext3DProviderWrapper> GetContextProviderWeakPtr()
      const {
    return webnn_control_client_->GetContextProviderWeakPtr();
  }
  const WebnnProcTable& GetProcs() const {
    return webnn_control_client_->GetProcs();
  }

  // Ensure commands up until now on this object's parent context are flushed by
  // the end of the task.
  void EnsureFlush();

  // Flush commands up until now on this object's parent context immediately.
  void FlushNow();

  // MLObjectBase mixin implementation
  const String& label() const { return label_; }
  void setLabel(const String& value);

 private:
  scoped_refptr<WebnnControlClientHolder> webnn_control_client_;
  String label_;
};

class WebnnObjectImpl : public ScriptWrappable, public WebnnObjectBase {
 public:
  explicit WebnnObjectImpl(MLContext* context);
  ~WebnnObjectImpl() override;

  WNNContext GetContextHandle();

  void Trace(Visitor* visitor) const override;

 protected:
  Member<MLContext> context_;
};

template <typename Handle>
class WebnnObject : public WebnnObjectImpl {
 public:
  WebnnObject(MLContext* context, Handle handle)
      : WebnnObjectImpl(context),
        handle_(handle),
        context_handle_(GetContextHandle()) {
    // All WebML Blink objects created directly or by the Context hold a
    // Member<MLContext> which keeps the context alive. However, this does not
    // enforce that the MLContext is finalized after all objects referencing it.
    // Add an extra ref in this constructor, and a release in the destructor to
    // ensure that the Webnn context is destroyed last.
    // TODO(enga): Investigate removing Member<MLContext>.
    GetProcs().contextReference(context_handle_);
  }

  ~WebnnObject() override {
    // Note: The context is released last because all child objects must be
    // destroyed first.
    (GetProcs().*WNNReleaseFn<Handle>::fn)(handle_);
    GetProcs().contextRelease(context_handle_);
  }

  Handle GetHandle() const { return handle_; }

 private:
  Handle const handle_;
  WNNContext context_handle_;
};

template <>
class WebnnObject<WNNContext> : public WebnnObjectBase {
 public:
  WebnnObject(scoped_refptr<WebnnControlClientHolder> webnn_control_client,
              WNNContext handle)
      : WebnnObjectBase(webnn_control_client), handle_(handle) {}
  ~WebnnObject() { GetProcs().contextRelease(handle_); }

  WNNContext GetHandle() const { return handle_; }

 private:
  WNNContext const handle_;
};

}  // namespace blink

#undef WEBNN_OBJECTS

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBNN_WEBNN_OBJECT_H_

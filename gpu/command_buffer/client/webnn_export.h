// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GPU_COMMAND_BUFFER_CLIENT_WEBNN_IMPLEMENTATION_EXPORT_H_
#define GPU_COMMAND_BUFFER_CLIENT_WEBNN_IMPLEMENTATION_EXPORT_H_

#if defined(COMPONENT_BUILD) && !defined(NACL_WIN64)
#if defined(WIN32)

#if defined(WEBNN_IMPLEMENTATION)
#define WEBNN_IMPLEMENTATION_EXPORT __declspec(dllexport)
#else
#define WEBNN_IMPLEMENTATION_EXPORT __declspec(dllimport)
#endif  // defined(WEBNN_IMPLEMENTATION)

#else  // defined(WIN32)
#if defined(WEBNN_IMPLEMENTATION)
#define WEBNN_IMPLEMENTATION_EXPORT __attribute__((visibility("default")))
#else
#define WEBNN_IMPLEMENTATION_EXPORT
#endif
#endif

#else  // defined(COMPONENT_BUILD)
#define WEBNN_IMPLEMENTATION_EXPORT
#endif

#endif  // GPU_COMMAND_BUFFER_CLIENT_WEBNN_IMPLEMENTATION_EXPORT_H_

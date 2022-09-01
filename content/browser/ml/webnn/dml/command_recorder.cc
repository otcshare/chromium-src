// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/command_recorder.h"

namespace content::webnn {

CommandRecorder::~CommandRecorder() = default;

CommandRecorder::CommandRecorder(ComPtr<IDMLDevice> dml_device)
    : dml_device_(dml_device) {}

HRESULT CommandRecorder::Initialize() {
  HRESULT hr = dml_device_->GetParentDevice(IID_PPV_ARGS(&d3d12_device_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = d3d12_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&command_allocator_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = d3d12_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        command_allocator_.Get(), nullptr,
                                        IID_PPV_ARGS(&command_list_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = dml_device_->CreateOperatorInitializer(
      0, nullptr, IID_PPV_ARGS(&operator_initializer_));
  if (FAILED(hr)) {
    return hr;
  }
  hr = dml_device_->CreateCommandRecorder(IID_PPV_ARGS(&command_recorder_));
  if (FAILED(hr)) {
    return hr;
  }
  return S_OK;
}

HRESULT CommandRecorder::InitializeOperator(
    IDMLCompiledOperator* compiled_operator,
    const std::vector<std::shared_ptr<InputEdgeInfo>>& inputs) {
  // Reset the initializer to reference the compiled operator.
  IDMLCompiledOperator* ops[] = {compiled_operator};
  HRESULT hr = operator_initializer_->Reset(ARRAYSIZE(ops), ops);
  if (FAILED(hr)) {
    return hr;
  }

  // TODO
  DML_BINDING_PROPERTIES initializeBindingProperties =
      operator_initializer_->GetBindingProperties();
  DML_BINDING_PROPERTIES executeBindingProperties =
      compiled_operator->GetBindingProperties();
  UINT descriptorCount =
      std::max(initializeBindingProperties.RequiredDescriptorCount,
               executeBindingProperties.RequiredDescriptorCount);

  // Describe and create a constant buffer view (CBV), Shader resource view
  // (SRV), and unordered access view (UAV) descriptor heap.
  D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
  descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  descriptorHeapDesc.NumDescriptors = descriptorCount;
  descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  hr = d3d12_device_->CreateDescriptorHeap(&descriptorHeapDesc,
                                           IID_PPV_ARGS(&mDescriptorHeap));
  if (FAILED(hr)) {
    return hr;
  }
  // Set the descriptor heap(s).
  ID3D12DescriptorHeap* descriptorHeaps[] = {mDescriptorHeap.Get()};
  command_list_->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                    descriptorHeaps);

  // Create a binding table over the descriptor heap we just created.
  mBindingTableDesc.Dispatchable = operator_initializer_.Get();
  mBindingTableDesc.CPUDescriptorHandle =
      mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
  mBindingTableDesc.GPUDescriptorHandle =
      mDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
  // The size of the binding table, in descriptors. This is the maximum number
  // of descriptors that DirectML is permitted to write, from the start of both
  // the supplied CPU and GPU descriptor handles.
  mBindingTableDesc.SizeInDescriptors = descriptorCount;

  hr = dml_device_->CreateBindingTable(&mBindingTableDesc,
                                       IID_PPV_ARGS(&mBindingTable));
  if (FAILED(hr)) {
    return hr;
  }
  mTemporaryResourceSize =
      std::max(initializeBindingProperties.TemporaryResourceSize,
               executeBindingProperties.TemporaryResourceSize);
  mPersistentResourceSize = executeBindingProperties.PersistentResourceSize;

  // Bind and initialize the operator on the GPU.
  if (mTemporaryResourceSize != 0) {
    D3D12_HEAP_PROPERTIES heap_properties = CreateHeapProperties();
    D3D12_RESOURCE_DESC resource_desc = CreateResourceDesc(
        mTemporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    d3d12_device_->CreateCommittedResource(
        &heap_properties, D3D12_HEAP_FLAG_NONE, &resource_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&mTemporaryResource));

    if (initializeBindingProperties.TemporaryResourceSize != 0) {
      DML_BUFFER_BINDING bufferBinding{mTemporaryResource.Get(), 0,
                                       mTemporaryResourceSize};
      DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
      mBindingTable->BindTemporaryResource(&bindingDesc);
    }
  }

  // Persistent resources must be supplied during initialization of a compiled
  // operator (where it is bound as an output of the operator initializer) as
  // well as during execution.
  if (mPersistentResourceSize != 0) {
    D3D12_HEAP_PROPERTIES heap_properties = CreateHeapProperties();
    D3D12_RESOURCE_DESC resource_desc = CreateResourceDesc(
        mPersistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    d3d12_device_->CreateCommittedResource(
        &heap_properties, D3D12_HEAP_FLAG_NONE, &resource_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&mPersistentResource));

    DML_BUFFER_BINDING bufferBinding{mPersistentResource.Get(), 0,
                                     mPersistentResourceSize};
    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
    mBindingTable->BindOutputs(1, &bindingDesc);
  }

  // Initialize constant inputs.
  uint64_t constantInputsResourceSize = 0;
  for (auto& input : inputs) {
    if (input->isConstantInput) {
      uint64_t offset =
          RoundUpToMultiple(constantInputsResourceSize,
                            (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
      constantInputsResourceSize = offset + input->byteLength;
    }
  }

  if (constantInputsResourceSize) {
    D3D12_HEAP_PROPERTIES upload_heap_properties =
        CreateHeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC upload_resource_desc =
        CreateResourceDesc(constantInputsResourceSize);
    hr = d3d12_device_->CreateCommittedResource(
        &upload_heap_properties, D3D12_HEAP_FLAG_NONE, &upload_resource_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&mUploadResource));
    if (FAILED(hr)) {
      return hr;
    }

    D3D12_HEAP_PROPERTIES input_heap_properties = CreateHeapProperties();
    D3D12_RESOURCE_DESC input_resource_desc = CreateResourceDesc(
        constantInputsResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    hr = d3d12_device_->CreateCommittedResource(
        &input_heap_properties, D3D12_HEAP_FLAG_NONE, &input_resource_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&mInputResource));
    if (FAILED(hr)) {
      return hr;
    }

    std::vector<DML_BUFFER_BINDING> inputBufferBinding(inputs.size());
    FillUploadResourceAndInputBindings(constantInputsResourceSize,
                                       inputBufferBinding, {}, inputs,
                                       mUploadResource, mInputResource);
    // Copy buffer from mUploadResource to mInputResource.
    CopyBufferRegion(command_list_, mUploadResource, mInputResource,
                     constantInputsResourceSize,
                     D3D12_RESOURCE_STATE_COPY_DEST);

    DML_BUFFER_ARRAY_BINDING inputBufferArrayBinding = {};
    inputBufferArrayBinding.BindingCount = inputBufferBinding.size();
    inputBufferArrayBinding.Bindings = inputBufferBinding.data();
    DML_BINDING_DESC inputBindingDesc{DML_BINDING_TYPE_BUFFER_ARRAY,
                                      &inputBufferArrayBinding};
    mBindingTable->BindInputs(1, &inputBindingDesc);
  }

  // Record execution of the operator initializer.
  // The command recorder is a stateless object that records Dispatches into an
  // existing Direct3D 12 command list.
  command_recorder_->RecordDispatch(
      command_list_.Get(), operator_initializer_.Get(), mBindingTable.Get());

  return S_OK;
}

HRESULT CommandRecorder::ExecuteOperator(
    IDMLCompiledOperator* compiled_operator,
    NamedInputsPtr namedInputs,
    const std::vector<std::shared_ptr<InputEdgeInfo>>& inputs,
    const std::vector<DML_BINDING_DESC>& output_bindings,
    UINT64 commonInputsResourceSize,
    ComPtr<ID3D12Resource> uploadResource,
    ComPtr<ID3D12Resource> inputResource) {
  // Bind and execute the operator on the GPU.
  // Reset the binding table to bind for the operator we want to execute (it
  // was previously used to bind for the initializer).
  mBindingTableDesc.Dispatchable = compiled_operator;
  mBindingTable->Reset(&mBindingTableDesc);

  if (mTemporaryResourceSize != 0) {
    DML_BUFFER_BINDING bufferBinding{mTemporaryResource.Get(), 0,
                                     mTemporaryResourceSize};
    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
    mBindingTable->BindTemporaryResource(&bindingDesc);
  }

  if (mPersistentResourceSize != 0) {
    DML_BUFFER_BINDING bufferBinding{mPersistentResource.Get(), 0,
                                     mPersistentResourceSize};
    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
    mBindingTable->BindPersistentResource(&bindingDesc);
  }

  // Initialize common inputs.
  if (commonInputsResourceSize) {
    std::vector<DML_BUFFER_BINDING> inputBufferBinding(inputs.size());
    FillUploadResourceAndInputBindings(
        commonInputsResourceSize, inputBufferBinding, std::move(namedInputs),
        inputs, uploadResource, inputResource);
    // Copy buffer from uploadResource to inputResource.
    CopyBufferRegion(command_list_, uploadResource, inputResource,
                     commonInputsResourceSize, D3D12_RESOURCE_STATE_COPY_DEST);

    std::vector<DML_BINDING_DESC> inputBindingDesc(inputs.size());
    for (size_t i = 0; i < inputBufferBinding.size(); ++i) {
      if (inputBufferBinding[i].Buffer != nullptr) {
        inputBindingDesc[i] = {DML_BINDING_TYPE_BUFFER, &inputBufferBinding[i]};
      }
    }
    mBindingTable->BindInputs(inputBindingDesc.size(), inputBindingDesc.data());
  }

  mBindingTable->BindOutputs(output_bindings.size(), output_bindings.data());

  // Record execution of the compiled operator.
  ID3D12DescriptorHeap* descriptorHeaps[] = {mDescriptorHeap.Get()};
  command_list_->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                    descriptorHeaps);
  command_recorder_->RecordDispatch(command_list_.Get(), compiled_operator,
                                    mBindingTable.Get());
  return S_OK;
}

void CommandRecorder::FillUploadResourceAndInputBindings(
    uint64_t uploadResourceSize,
    std::vector<DML_BUFFER_BINDING>& inputBufferBinding,
    NamedInputsPtr namedInputs,
    const std::vector<std::shared_ptr<InputEdgeInfo>>& inputs,
    ComPtr<ID3D12Resource> uploadResource,
    ComPtr<ID3D12Resource> inputResource) {
  D3D12_RANGE uploadBufferRange{0, uploadResourceSize};
  int8_t* uploadBuffer;
  WEBNN_CHECK(uploadResource->Map(0, &uploadBufferRange,
                                  reinterpret_cast<void**>(&uploadBuffer)));
  uint64_t offset = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (namedInputs.get() == nullptr) {
      if (input->isConstantInput) {
        offset = RoundUpToMultiple(
            offset, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
        inputBufferBinding[i].Buffer = inputResource.Get();
        inputBufferBinding[i].Offset = offset;
        inputBufferBinding[i].SizeInBytes = input->byteLength;
        memcpy(uploadBuffer + offset, input->buffer,
               static_cast<size_t>(input->byteLength));
        offset = offset + input->byteLength;
      }
    } else {
      if (!input->isConstantInput) {
        offset = RoundUpToMultiple(
            offset, (uint64_t)DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
        ml::webnn::mojom::MemoryInfoPtr memory_info =
            std::move(namedInputs->inputs[input->name]);
        base::ReadOnlySharedMemoryRegion& shared_memory_region =
            namedInputs->shared_memory;
        DCHECK(shared_memory_region.IsValid());
        base::ReadOnlySharedMemoryMapping shared_memory_mapping =
            shared_memory_region.MapAt(memory_info->byte_offset,
                                       memory_info->byte_length);
        inputBufferBinding[i].Buffer = inputResource.Get();
        inputBufferBinding[i].Offset = offset;
        inputBufferBinding[i].SizeInBytes = memory_info->byte_length;
        memcpy(uploadBuffer + offset,
               shared_memory_mapping.GetMemoryAs<uint8_t>(),
               memory_info->byte_length);
        offset = offset + memory_info->byte_length;
      }
    }
  }
  uploadResource->Unmap(0, nullptr);
}

ComPtr<ID3D12CommandAllocator> CommandRecorder::GetCommandAllocator() {
  return command_allocator_;
}

ComPtr<ID3D12GraphicsCommandList> CommandRecorder::GetCommandList() {
  return command_list_;
}

ComPtr<IDMLDevice> CommandRecorder::GetDMLDevice() {
  return dml_device_;
}

}  // namespace content::webnn

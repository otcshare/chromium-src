// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef INCLUDE_MIN_GPGMM_H_
#define INCLUDE_MIN_GPGMM_H_

#include <cstdint>
#include <memory>

namespace gpgmm {

class MemoryBase {
 public:
  MemoryBase(uint64_t size, uint64_t alignment);
  virtual ~MemoryBase();

  uint64_t GetSize() const;
  uint64_t GetAlignment() const;

 private:
  const uint64_t mSize;
  const uint64_t mAlignment;
};

struct MemoryAllocatorInfo {
  uint32_t UsedBlockCount;
  uint64_t UsedBlockUsage;
  uint32_t UsedMemoryCount;
  uint64_t UsedMemoryUsage;
  uint64_t FreeMemoryUsage;
  uint64_t PrefetchedMemoryMisses;
  uint64_t PrefetchedMemoryMissesEliminated;
  uint64_t SizeCacheMisses;
  uint64_t SizeCacheHits;
};

class MemoryAllocation;

class MemoryAllocator {
 public:
  virtual void DeallocateMemory(
      std::unique_ptr<MemoryAllocation> allocation) = 0;
  virtual uint64_t ReleaseMemory(uint64_t bytesToRelease);
  virtual MemoryAllocatorInfo GetInfo() const;

 protected:
  MemoryAllocatorInfo mInfo = {};
};

struct MemoryAllocationInfo {
  uint64_t SizeInBytes;
  uint64_t Alignment;
};

enum AllocationMethod {
  kUndefined = 0,
  kStandalone = 1,
  kSubAllocated = 2,
  kSubAllocatedWithin = 3,
};

struct MemoryBlock {
  uint64_t Offset;
  uint64_t Size;
};

class MemoryAllocation {
 public:
  MemoryAllocation(MemoryAllocator* allocator,
                   MemoryBase* memory,
                   uint64_t requestSize);

  virtual ~MemoryAllocation();

  MemoryAllocation(const MemoryAllocation&);
  MemoryAllocation& operator=(const MemoryAllocation&);
  bool operator==(const MemoryAllocation&) const;
  bool operator!=(const MemoryAllocation& other) const;

  MemoryAllocationInfo GetInfo() const;
  MemoryBase* GetMemory() const;
  uint8_t* GetMappedPointer() const;
  MemoryAllocator* GetAllocator() const;
  uint64_t GetSize() const;
  uint64_t GetRequestSize() const;
  uint64_t GetAlignment() const;
  uint64_t GetOffset() const;
  AllocationMethod GetMethod() const;
  MemoryBlock* GetBlock() const;

 protected:
  MemoryAllocator* mAllocator;

 private:
  MemoryBase* mMemory;
  uint64_t mRequestSize;
};

}  // namespace gpgmm

#endif  // INCLUDE_MIN_GPGMM_H_

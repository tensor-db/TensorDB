#include "native.h"

#include <atomic>

namespace spectradb_native {

namespace {
std::atomic<std::uint64_t> g_hash_calls{0};
}  // namespace

std::uint64_t native_hash64(rust::Slice<const std::uint8_t> bytes) {
  g_hash_calls.fetch_add(1, std::memory_order_relaxed);

  std::uint64_t h = 0xcbf29ce484222325ull;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    h ^= static_cast<std::uint64_t>(bytes[i]);
    h *= 0x100000001b3ull;
    h ^= (h >> 32);
  }
  return h;
}

std::uint64_t native_hash_call_count() { return g_hash_calls.load(std::memory_order_relaxed); }

void native_hash_reset_count() { g_hash_calls.store(0, std::memory_order_relaxed); }

}  // namespace spectradb_native

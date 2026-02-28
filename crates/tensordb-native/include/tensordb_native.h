#pragma once

#include <cstdint>
#include "rust/cxx.h"

namespace tensordb_native {

std::uint64_t native_hash64(rust::Slice<const std::uint8_t> bytes);
std::uint64_t native_hash_call_count();
void native_hash_reset_count();

}  // namespace tensordb_native

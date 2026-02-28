#!/usr/bin/env bash
# Convert the merged HuggingFace model to GGUF Q8_0 format.
#
# Prerequisites:
#   1. Run finetune_nl2sql.py first (produces merged model in scripts/output/tensordb-nl2sql/merged/)
#   2. llama.cpp repo at .local/llama.cpp/
#
# Usage:
#   bash scripts/convert_to_gguf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_CPP_DIR="$PROJECT_DIR/.local/llama.cpp"
MERGED_DIR="$SCRIPT_DIR/output/tensordb-nl2sql/merged"
OUTPUT_DIR="$SCRIPT_DIR/output/tensordb-nl2sql"
MODEL_DIR="$PROJECT_DIR/.local/models"

F16_GGUF="$OUTPUT_DIR/Qwen3-0.6B-f16.gguf"
Q8_GGUF="$OUTPUT_DIR/Qwen3-0.6B-Q8_0.gguf"

# ---------- Validate ----------
if [ ! -d "$MERGED_DIR" ]; then
    echo "ERROR: Merged model not found at $MERGED_DIR"
    echo "Run finetune_nl2sql.py first."
    exit 1
fi

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "ERROR: llama.cpp not found at $LLAMA_CPP_DIR"
    exit 1
fi

# ---------- Step 1: Build llama-quantize if needed ----------
QUANTIZE_BIN="$LLAMA_CPP_DIR/build-quantize/bin/llama-quantize"
if [ ! -f "$QUANTIZE_BIN" ]; then
    echo "==> Building llama-quantize..."
    cd "$LLAMA_CPP_DIR"
    cmake -B build-quantize \
        -DGGML_CUDA=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=OFF
    cmake --build build-quantize --target llama-quantize -j"$(nproc)"
    cd "$PROJECT_DIR"
    echo "  Built: $QUANTIZE_BIN"
else
    echo "==> llama-quantize already built: $QUANTIZE_BIN"
fi

# ---------- Step 2: Convert HF → GGUF f16 ----------
CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "ERROR: convert_hf_to_gguf.py not found at $CONVERT_SCRIPT"
    exit 1
fi

echo "==> Converting HuggingFace model → GGUF f16..."
conda run -n cortex_ngc python "$CONVERT_SCRIPT" \
    "$MERGED_DIR" \
    --outtype f16 \
    --outfile "$F16_GGUF"
echo "  Wrote: $F16_GGUF"

# ---------- Step 3: Quantize f16 → Q8_0 ----------
echo "==> Quantizing f16 → Q8_0..."
"$QUANTIZE_BIN" "$F16_GGUF" "$Q8_GGUF" Q8_0
echo "  Wrote: $Q8_GGUF"

# ---------- Step 4: Clean up f16 (optional, it's large) ----------
rm -f "$F16_GGUF"
echo "  Removed intermediate f16 file."

# ---------- Summary ----------
echo ""
echo "============================================"
echo "  Fine-tuned GGUF ready!"
echo "  Path: $Q8_GGUF"
echo "  Size: $(du -h "$Q8_GGUF" | cut -f1)"
echo "============================================"
echo ""
echo "To deploy:"
echo "  1. Backup:  cp $MODEL_DIR/Qwen3-0.6B-Q8_0.gguf $MODEL_DIR/Qwen3-0.6B-Q8_0.gguf.original"
echo "  2. Replace: cp $Q8_GGUF $MODEL_DIR/Qwen3-0.6B-Q8_0.gguf"
echo "  3. Rebuild: maturin develop --release"
echo ""

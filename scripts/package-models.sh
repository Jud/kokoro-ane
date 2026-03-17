#!/bin/bash
set -euo pipefail

# Package essential model files into a tarball for GitHub Release upload.
# Usage: ./scripts/package-models.sh <model-dir> [output-path]

MODEL_DIR="${1:?Usage: $0 <model-dir> [output-path]}"
OUTPUT="${2:-kokoro-models.tar.gz}"

if [ ! -d "$MODEL_DIR/voices" ]; then
    echo "Error: $MODEL_DIR does not look like a kokoro model directory (no voices/)"
    exit 1
fi

echo "Packaging models from: $MODEL_DIR"

cd "$MODEL_DIR"

tar czf "$OLDPWD/$OUTPUT" \
    kokoro_21_5s_frontend.mlmodelc \
    kokoro_21_5s_backend.mlmodelc \
    kokoro_24_10s_frontend.mlmodelc \
    kokoro_24_10s_backend.mlmodelc \
    voices

cd "$OLDPWD"
SIZE=$(du -h "$OUTPUT" | cut -f1)
echo "Created $OUTPUT ($SIZE)"
echo ""
echo "Upload with:"
echo "  gh release create models-\$(date +%Y-%m-%d) $OUTPUT --repo Jud/kokoro-tts-swift --title 'Models (split pipeline)' --notes 'Kokoro-82M split CoreML models and voice embeddings'"

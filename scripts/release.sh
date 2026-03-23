#!/bin/bash
set -euo pipefail

# Release script for KokoroCoreML CoreML models.
#
# Exports PyTorch → CoreML (production, non-deterministic), palettizes
# weights (8-bit), converts voices to binary, compiles, packages, and
# uploads as a GitHub release.
#
# Usage:
#   ./scripts/release.sh              # tag: models-YYYY-MM-DD
#   ./scripts/release.sh --dry-run    # export + package without uploading

REPO="Jud/kokoro-coreml"
TAG="models-$(date +%Y-%m-%d)"
DRY_RUN="${1:-}"
EXPORT_DIR="models_export"
TARBALL="kokoro-models.tar.gz"

echo "=== KokoroCoreML Release: $TAG ==="

# 1. Export CoreML models (production — natural randomness, then palettize)
echo ""
echo "Step 1: Exporting CoreML models (production)..."
PYTHONPATH=scripts .venv/bin/python scripts/export_coreml.py --output-dir "$EXPORT_DIR" --skip-palettize

# 2. Palettize weights (8-bit, ~75% size reduction)
echo ""
echo "Step 2: Palettizing weights..."
PYTHONPATH=scripts .venv/bin/python -c "
from export_coreml import _export_palettized
_export_palettized('$EXPORT_DIR')
"

# 3. Compile palettized models to .mlmodelc
echo ""
echo "Step 3: Compiling palettized models..."
for name in kokoro_frontend_pal8 kokoro_backend_pal8; do
    pkg="$EXPORT_DIR/$name.mlpackage"
    if [ ! -d "$pkg" ]; then
        echo "    ✗ $pkg not found"
        exit 1
    fi
    echo "  Compiling $name..."
    xcrun coremlcompiler compile "$pkg" "$EXPORT_DIR"
    if [ -d "$EXPORT_DIR/$name.mlmodelc" ]; then
        echo "    ✓ $name.mlmodelc"
    else
        echo "    ✗ Compilation failed for $name"
        exit 1
    fi
done

# 4. Voice data — convert JSON to binary if needed
VOICE_DIR="$EXPORT_DIR/voices"
if [ ! -d "$VOICE_DIR" ]; then
    # Look for JSON voices in the model directory
    JSON_VOICE_DIR="$HOME/Library/Application Support/kokoro-coreml/models/kokoro/voices"
    if [ ! -d "$JSON_VOICE_DIR" ]; then
        JSON_VOICE_DIR="$HOME/Library/Application Support/kokoro-tts/models/kokoro/voices"
    fi
    if [ -d "$JSON_VOICE_DIR" ]; then
        echo ""
        echo "Step 4: Converting voice embeddings to binary..."
        PYTHONPATH=scripts .venv/bin/python -c "
from export_coreml import convert_voices_to_binary
convert_voices_to_binary('$JSON_VOICE_DIR', '$VOICE_DIR')
"
    else
        echo ""
        echo "Step 4: Voice data not found"
        echo "  Copy voice embeddings from your model directory:"
        echo "    cp -r ~/Library/Application\\ Support/kokoro-coreml/models/kokoro/voices $EXPORT_DIR/"
        exit 1
    fi
else
    # If voices dir exists but has JSON files, convert them
    if ls "$VOICE_DIR"/*.json >/dev/null 2>&1; then
        echo ""
        echo "Step 4: Converting voice embeddings to binary..."
        PYTHONPATH=scripts .venv/bin/python -c "
from export_coreml import convert_voices_to_binary
convert_voices_to_binary('$VOICE_DIR', '$VOICE_DIR')
"
        rm -f "$VOICE_DIR"/*.json
    else
        echo ""
        echo "Step 4: Voice data present ($(ls "$VOICE_DIR"/*.bin 2>/dev/null | wc -l | tr -d ' ') voices)"
    fi
fi

# 5. Package (palettized models + binary voices)
echo ""
echo "Step 5: Packaging..."
cd "$EXPORT_DIR"
# Replace float32 models with palettized versions
rm -rf kokoro_frontend.mlmodelc kokoro_backend.mlmodelc
mv kokoro_frontend_pal8.mlmodelc kokoro_frontend.mlmodelc
mv kokoro_backend_pal8.mlmodelc kokoro_backend.mlmodelc
tar czf "../$TARBALL" \
    kokoro_frontend.mlmodelc \
    kokoro_backend.mlmodelc \
    voices
cd ..

SIZE=$(du -h "$TARBALL" | cut -f1)
echo "  Created $TARBALL ($SIZE)"

# 6. Upload
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo ""
    echo "Dry run — skipping upload. To upload manually:"
    echo "  gh release create $TAG $TARBALL --repo $REPO --title 'Models ($TAG)'"
else
    echo ""
    echo "Step 6: Uploading to GitHub release $TAG..."
    gh release create "$TAG" "$TARBALL" \
        --repo "$REPO" \
        --title "Models ($TAG)" \
        --notes "Kokoro-82M CoreML models (8-bit palettized, production) and voice embeddings (float32 binary)."
    echo "  ✓ https://github.com/$REPO/releases/tag/$TAG"
fi

echo ""
echo "Done."

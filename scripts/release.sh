#!/bin/bash
set -euo pipefail

# Release script for KokoroTTS CoreML models.
#
# Exports PyTorch → CoreML, compiles, packages with voice data,
# and uploads as a GitHub release with a date-based tag.
#
# Usage:
#   ./scripts/release.sh              # tag: models-YYYY-MM-DD
#   ./scripts/release.sh --dry-run    # export + package without uploading

REPO="Jud/kokoro-tts-swift"
TAG="models-$(date +%Y-%m-%d)"
DRY_RUN="${1:-}"
EXPORT_DIR="models_export"
TARBALL="kokoro-models.tar.gz"

echo "=== KokoroTTS Release: $TAG ==="

# 1. Export CoreML models
echo ""
echo "Step 1: Exporting CoreML models..."
.venv/bin/python scripts/export_coreml.py --output-dir "$EXPORT_DIR"

# 2. Compile to .mlmodelc
echo ""
echo "Step 2: Compiling models..."
for pkg in "$EXPORT_DIR"/*.mlpackage; do
    name=$(basename "$pkg" .mlpackage)
    echo "  Compiling $name..."
    xcrun coremlcompiler compile "$pkg" "$EXPORT_DIR"
    if [ -d "$EXPORT_DIR/$name.mlmodelc" ]; then
        echo "    ✓ $name.mlmodelc"
    else
        echo "    ✗ Compilation failed for $name"
        exit 1
    fi
done

# 3. Download voice data if not present
VOICE_DIR="$EXPORT_DIR/voices"
if [ ! -d "$VOICE_DIR" ]; then
    echo ""
    echo "Step 3: Voice data not found in $EXPORT_DIR/voices"
    echo "  Copy voice embeddings from your model directory:"
    echo "    cp -r ~/Library/Application\\ Support/kokoro-tts/models/kokoro/voices $EXPORT_DIR/"
    exit 1
else
    echo ""
    echo "Step 3: Voice data present ($(ls "$VOICE_DIR" | wc -l | tr -d ' ') voices)"
fi

# 4. Package
echo ""
echo "Step 4: Packaging..."
cd "$EXPORT_DIR"
tar czf "../$TARBALL" \
    kokoro_21_5s_frontend.mlmodelc \
    kokoro_21_5s_backend.mlmodelc \
    kokoro_24_10s_frontend.mlmodelc \
    kokoro_24_10s_backend.mlmodelc \
    voices
cd ..

SIZE=$(du -h "$TARBALL" | cut -f1)
echo "  Created $TARBALL ($SIZE)"

# 5. Upload
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo ""
    echo "Dry run — skipping upload. To upload manually:"
    echo "  gh release create $TAG $TARBALL --repo $REPO --title 'Models ($TAG)'"
else
    echo ""
    echo "Step 5: Uploading to GitHub release $TAG..."
    gh release create "$TAG" "$TARBALL" \
        --repo "$REPO" \
        --title "Models ($TAG)" \
        --notes "Kokoro-82M CoreML models and voice embeddings."
    echo "  ✓ https://github.com/$REPO/releases/tag/$TAG"
fi

echo ""
echo "Done."

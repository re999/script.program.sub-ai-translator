#!/bin/bash
set -e

# Katalog skryptu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_DIR="$SCRIPT_DIR"
ADDON_XML="$ADDON_DIR/addon.xml"
DIST_DIR="$SCRIPT_DIR/dist"

# Ustal ID dodatku (czytaj z addon.xml)
ADDON_ID=$(xmllint --xpath 'string(/addon/@id)' "$ADDON_XML")

# Wydobądź wersję
version=$(xmllint --xpath 'string(/addon/@version)' "$ADDON_XML")

if [[ -z "$version" ]]; then
  echo "❌ Nie udało się znaleźć wersji w $ADDON_XML"
  exit 1
fi

# Podbij wersję patch (X.Y.Z → X.Y.(Z+1))
IFS='.' read -r major minor patch <<< "$version"
new_patch=$((patch + 1))
new_version="${major}.${minor}.${new_patch}"

# Zastąp wersję w addon.xml
tmp_file=$(mktemp)
xmllint --xpath "/*" "$ADDON_XML" | sed "s/version=\"$version\"/version=\"$new_version\"/" > "$tmp_file"
mv "$tmp_file" "$ADDON_XML"

# Utwórz katalog na ZIP-y
mkdir -p "$DIST_DIR"

# Nazwa ZIP-a
zip_name="${ADDON_ID}-${new_version}.zip"
zip_path="${DIST_DIR}/${zip_name}"

# Stwórz ZIP z całym katalogiem dodatku (wymagane przez Kodi)
cd "$SCRIPT_DIR/.."
zip -r "$zip_path" "$(basename "$SCRIPT_DIR")" \
  -x "*.pyc" "*__pycache__/*" "*.DS_Store" "*.git*" "*wheels/*" "*/dist/*" "*/build.sh" "*.pytest*" \
  "*/.gitignore" "*/.git" "*/tests/*" "*/tests" "*/test.py" "*/test.sh" "*/.venv/*" ".venv/*"

echo "✅ Zbudowano: $zip_path (wersja: $new_version)"
echo "🔄 Zaktualizowano addon.xml do wersji: $new_version"

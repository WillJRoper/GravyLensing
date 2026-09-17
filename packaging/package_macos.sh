#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BUILD_DIR="${BUILD_DIR:-$ROOT/build-release}"
DIST_DIR="${DIST_DIR:-$ROOT/dist}"
APP="$BUILD_DIR/GravyLensing.app"
RELEASE_VERSION="${VERSION:-1.0.0}"
RELEASE_VERSION="${RELEASE_VERSION#v}"
DMG="$DIST_DIR/GravyLensing-$RELEASE_VERSION-arm64.dmg"
QT_PREFIX="${QT_PREFIX:-$(brew --prefix qt)}"
MACDEPLOYQT="${MACDEPLOYQT:-$(brew --prefix qtbase)/bin/macdeployqt}"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_PREFIX_PATH="$QT_PREFIX;$(brew --prefix libomp)" \
  -DGRAVY_VERSION="$RELEASE_VERSION" \
  -DBUILD_TESTS=OFF
rm -rf "$APP"
cmake --build "$BUILD_DIR" --config Release --parallel

COCOA_PLUGIN="$APP/Contents/PlugIns/platforms/libqcocoa.dylib"
mkdir -p "$(dirname "$COCOA_PLUGIN")"
cp "$(brew --prefix qtbase)/share/qt/plugins/platforms/libqcocoa.dylib" \
  "$COCOA_PLUGIN"
"$MACDEPLOYQT" "$APP" -always-overwrite -verbose=1 -no-codesign \
  -no-plugins -executable="$COCOA_PLUGIN" -libpath="$QT_PREFIX/lib"

# macdeployqt only writes qt.conf when it deploys plugins itself; without it Qt
# keeps the Homebrew plugin dir in its search path and loads that libqcocoa
# instead of the bundled one, which aborts on a different-Team-ID signature.
printf '[Paths]\nPlugins = PlugIns\n' > "$APP/Contents/Resources/qt.conf"

# Homebrew's dylibs carry an LC_RPATH pointing back into their Cellar.
# macdeployqt rewrites dependencies to @rpath but leaves those rpaths in
# place, so on any machine that also has the same formulae installed dyld
# resolves @rpath against Homebrew first and loads a second copy of OpenCV,
# FFmpeg and libomp - which aborts at startup with OpenMP error #15.
while IFS= read -r -d '' file; do
  otool -l "$file" 2>/dev/null |
    awk '/LC_RPATH/ {rpath = 1} rpath && $1 == "path" {print $2; rpath = 0}' |
    grep -E '^/(opt/homebrew|usr/local)/' |
    while IFS= read -r rpath; do
      install_name_tool -delete_rpath "$rpath" "$file"
    done || true  # grep exits 1 for the many files with nothing to strip
done < <(find "$APP/Contents" -type f -print0)

# macdeployqt leaves some install names (OpenCV, the Qt frameworks) pointing
# at Homebrew. Dependencies are rewritten to @rpath, but anything that dlopens
# a library by its install name would still reach outside the bundle, so point
# every ID back at the bundled copy.
while IFS= read -r -d '' file; do
  id=$(otool -D "$file" 2>/dev/null | sed -n '2p')
  case "$id" in
    /opt/homebrew/* | /usr/local/*)
      install_name_tool -id "@rpath/${file#"$APP/Contents/Frameworks/"}" "$file"
      ;;
  esac
done < <(find "$APP/Contents/Frameworks" -type f -print0)

# Nothing in the bundle may reference a path outside it. Dependencies and
# rpaths are both checked, for every Mach-O file - not only the ones that
# happen to carry an executable bit, which most bundled dylibs do not.
leaked=0
while IFS= read -r -d '' file; do
  refs=$(otool -l "$file" 2>/dev/null |
    awk '/^ *cmd LC_(ID_DYLIB|LOAD_DYLIB|LOAD_WEAK_DYLIB|REEXPORT_DYLIB|RPATH)$/ {want = 1}
         want && ($1 == "name" || $1 == "path") {print $2; want = 0}' |
    grep -E '^/(opt/homebrew|usr/local)/' || true)
  if [[ -n "$refs" ]]; then
    echo "Unresolved Homebrew reference in ${file#"$APP/"}:" >&2
    echo "$refs" >&2
    leaked=1
  fi
done < <(find "$APP/Contents" -type f -print0)
if (( leaked )); then
  echo "Package contains unresolved Homebrew dependencies" >&2
  exit 1
fi
if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --timestamp \
    --sign "$APPLE_SIGNING_IDENTITY" "$APP"
  codesign --force --options runtime --timestamp \
    --entitlements "$HERE/entitlements.plist" \
    --sign "$APPLE_SIGNING_IDENTITY" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict --verbose=2 "$APP"

rm -rf "$DIST_DIR/stage"
mkdir -p "$DIST_DIR/stage"
cp -R "$APP" "$DIST_DIR/stage/GravyLensing.app"
ln -s /Applications "$DIST_DIR/stage/Applications"
hdiutil create -volname GravyLensing -srcfolder "$DIST_DIR/stage" \
  -ov -format UDZO "$DMG"
rm -rf "$DIST_DIR/stage"

echo "$DMG"

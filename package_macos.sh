#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
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

if find "$APP/Contents" -type f -perm -111 -exec otool -L {} \; 2>/dev/null | \
    grep -E '/(opt/homebrew|usr/local)/' >/dev/null; then
  echo "Package contains unresolved Homebrew dependencies" >&2
  exit 1
fi
if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --timestamp \
    --sign "$APPLE_SIGNING_IDENTITY" "$APP"
  codesign --force --options runtime --timestamp \
    --entitlements "$ROOT/entitlements.plist" \
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

if [[ -n "${APPLE_NOTARY_PROFILE:-}" ]]; then
  xcrun notarytool submit "$DMG" \
    --keychain-profile "$APPLE_NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
fi

echo "$DMG"

# Packaging

macOS release packaging for GravyLensing.

| File | Purpose |
| --- | --- |
| `package_macos.sh` | Builds a Release `.app`, bundles Qt, signs it, and produces a DMG in `dist/`. |
| `entitlements.plist` | Hardened-runtime entitlements applied when signing with a Developer ID. |

## What the script does

1. Configures and builds an arm64 Release tree (deployment target 12.0, tests off).
2. Copies the Qt cocoa platform plugin into the bundle, then runs `macdeployqt`
   to vendor the Qt frameworks.
3. Fails if any bundled executable still links against a Homebrew prefix
   (`/opt/homebrew` or `/usr/local`) — that would break on other machines.
4. Signs the bundle, verifies the signature, and builds a compressed DMG.
5. Prints the DMG path on stdout.

## Local use

```sh
bash packaging/package_macos.sh
```

With no `APPLE_SIGNING_IDENTITY` set, the bundle is ad-hoc signed (`codesign -s -`).
That is fine for testing locally, but such a build is not distributable — other
machines will refuse to open it.

To produce a distributable build locally, set a Developer ID identity:

```sh
APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
  bash packaging/package_macos.sh
```

The script does not notarize. Notarization happens in CI (see below); to do it
by hand, run `xcrun notarytool submit <dmg> --wait` and `xcrun stapler staple <dmg>`
against the DMG the script prints.

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `VERSION` | `1.0.0` | Release version; a leading `v` is stripped. Sets `GRAVY_VERSION` and the DMG name. |
| `APPLE_SIGNING_IDENTITY` | unset | Developer ID identity. Unset means ad-hoc signing. |
| `BUILD_DIR` | `<repo>/build-release` | CMake build tree. |
| `DIST_DIR` | `<repo>/dist` | Where the DMG is written. |
| `QT_PREFIX` | `$(brew --prefix qt)` | Qt prefix used for `CMAKE_PREFIX_PATH` and `-libpath`. |
| `MACDEPLOYQT` | `$(brew --prefix qtbase)/bin/macdeployqt` | `macdeployqt` binary. |

## CI

`.github/workflows/release-macos.yml` runs this script on `v*` tags and on
manual dispatch. The workflow imports the Developer ID certificate from repository
secrets, exports `APPLE_SIGNING_IDENTITY`, then notarizes and staples the DMG
before uploading it and publishing the GitHub release.

Required repository secrets: `APPLE_CERTIFICATE_BASE64`,
`APPLE_CERTIFICATE_PASSWORD`, `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD`.

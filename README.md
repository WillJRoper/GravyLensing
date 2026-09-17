# GravyLensing

[![Docs](https://github.com/WillJRoper/gravy-lensing/actions/workflows/docs.yml/badge.svg)](https://willjroper.github.io/gravy-lensing/)
[![macOS Release](https://github.com/WillJRoper/gravy-lensing/actions/workflows/release-macos.yml/badge.svg)](https://github.com/WillJRoper/gravy-lensing/actions/workflows/release-macos.yml)
[![Latest Release](https://img.shields.io/github/v/release/WillJRoper/GravyLensing)](https://github.com/WillJRoper/GravyLensing/releases/latest)
[![License: GPLv3](https://img.shields.io/github/license/WillJRoper/GravyLensing)](LICENSE)
[![Downloads](https://img.shields.io/github/downloads/WillJRoper/GravyLensing/total)](https://github.com/WillJRoper/GravyLensing/releases)

A real-time gravitational lensing demo written in C++, supported by the
Goodwood Festival of Speed Future Lab.

GravyLensing takes a live camera feed, works out where the "mass" is in each
frame — a person, or an object of a chosen colour — and uses an FFT-based
deflection calculation to lens a background image around it, at camera frame
rate.

![SCR-20250429-qemj](https://github.com/user-attachments/assets/39f96883-53b8-4d13-b399-3d390bf4328f)

## Install

Apple Silicon users can download the latest DMG from
[GitHub Releases](https://github.com/WillJRoper/gravy-lensing/releases), drag
**GravyLensing** to **Applications**, and open it. Nothing else is required.

To build from source on macOS:

```bash
brew install cmake fftw libomp opencv qt
cmake -B build \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qtbase;/opt/homebrew/opt/libomp" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

This produces `build/GravyLensing.app`.

## Documentation

**[willjroper.github.io/gravy-lensing](https://willjroper.github.io/gravy-lensing/)**

- [Installation](https://willjroper.github.io/gravy-lensing/getting_started/installation.html)
  and [quick start](https://willjroper.github.io/gravy-lensing/getting_started/quickstart.html)
- [Ways to run the demo](https://willjroper.github.io/gravy-lensing/running_modes.html)
  — mirror mode and telescope mode, and how to choose
- [Functionality](https://willjroper.github.io/gravy-lensing/usage/index.html) —
  lens modes, masking, the lens itself, backgrounds, shortcuts, and CLI flags
- [Running a good demo](https://willjroper.github.io/gravy-lensing/recommendations.html)
  — setting up a real space
- [FAQ](https://willjroper.github.io/gravy-lensing/faq.html)

The sources live in [`docs/`](docs). Build them locally with:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

## Contributing

Contributions, issues, and feature requests are welcome. Fork the repository
and open a pull request.

The signed macOS DMG is built by `packaging/package_macos.sh`; see
[packaging/README.md](packaging/README.md).

## License

GNU GPL-3.0. See [LICENSE](LICENSE) for details.

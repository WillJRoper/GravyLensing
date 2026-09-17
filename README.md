# GravyLensing

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

Full documentation lives in [`docs/`](docs) and covers:

- [Installation](docs/source/getting_started/installation.rst) and
  [quick start](docs/source/getting_started/quickstart.rst)
- [Ways to run the demo](docs/source/running_modes.rst) — mirror mode and
  telescope mode, and how to choose
- [Functionality](docs/source/usage/index.rst) — lens modes, masking, the lens
  itself, backgrounds, shortcuts, and CLI flags
- [Running a good demo](docs/source/recommendations.rst) — setting up a real
  space
- [FAQ](docs/source/faq.rst)

Build it locally with:

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

Backgrounds
===========

The background image is the distant source being lensed. The macOS app ships
with a default set; any directory of supported images can be used instead.

Choosing and cycling
--------------------

- :kbd:`Left` and :kbd:`Right` step through the images in the directory.
- **Background > Automatic Cycling** advances them on a timer, with the
  interval set in **Background > Cycle Every** or in settings. An interval of
  ``-1`` means manual only.
- To use your own, open **Session > Settings...**, choose a backgrounds
  directory, and restart the session. Supported images in that directory are
  discovered automatically. **Restore Defaults** switches back to the packaged
  set.

Resolution and fitting
----------------------

Background resolution directly drives FFT memory use and frame rate, so it is
the setting to reach for first when performance is poor. Before a session
starts, every background is resized and cached at the selected output size;
that cache is why the first start after changing this takes a moment.

**Processing size** offers presets from 640×360 up to 4K, plus an explicit
custom width and height. 1080p at 30 fps is the recommended starting point.
Lower the resolution before lowering the frame rate — a smaller image at 30 fps
reads far better than a large one at 12.

**Image fitting** controls how an image that does not match the output aspect
ratio is handled:

- **Crop to Fill** — no bars, edges of the image are lost. Recommended.
- **Fit with Bars** — the whole image is visible, with bars at the sides.
- **Stretch to Fill** — fills the frame, may distort.

The **Cache** row on the Backgrounds page reports the processed-cache size and
can rebuild it, which is the fix if cached images ever look stale or wrong.

Choosing good images
--------------------

Lensing is only visible where there is structure to deflect. Images with
strong, recognisable features — a field of galaxies, a grid, sharp edges — show
the effect clearly. Smooth gradients and near-uniform images lens just as
correctly but look almost unchanged, which is a poor demo even though nothing
is broken. The packaged backgrounds include a grid overlay for exactly this
reason: it makes the deflection legible at a glance.

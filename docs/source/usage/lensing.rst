The lens
========

Once a mask exists, it is treated as a projected mass distribution. The
deflection field is built from it by convolution with a softened kernel,
evaluated with FFTs, and used to resample the background image. The result is
the deflection you would see looking through that mass — not a decorative warp.

The controls live under **Lens** in settings.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Control
     - What it does
   * - **Effect strength**
     - The deflection multiplier. How hard the background bends. Turning this
       up indefinitely does not make the demo better — past a point the
       background simply smears.
   * - **Effect width**
     - The kernel softening radius, in pixels. Small values give a tight,
       sharp distortion concentrated at the silhouette; large values spread
       the effect out into a broad, gentle bend.
   * - **Lens edge softness**
     - Smoothing applied at the mask boundary, independent of detection
       quality. Raise it when the silhouette edge looks hard or jagged.
   * - **Lens appearance**
     - Whether the interior of the mask is distorted too, not just the
       surroundings. On by default.
   * - **Lens contents**
     - Composites the real camera pixels inside the mask, so the subject
       appears in the output rather than a silhouette. Toggle live with
       :kbd:`Shift+I` or **View > Show Camera Inside Lens**.

Performance controls
--------------------

Under **Performance**:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Control
     - What it does
   * - **Worker threads**
     - Threads for the lensing stage. Defaults to your core count minus two,
       which deliberately leaves headroom for capture and display.
   * - **Calculation scale**
     - The internal resolution at which the mask and FFT are computed, as a
       fraction of the output size, before the result is upscaled. The single
       biggest performance lever.
   * - **Edge protection**
     - FFT padding factor: Standard, Extra, Maximum, or a custom value.
       Increase it if you see wrap-around artefacts at the frame edges, where
       the deflection appears to re-enter from the opposite side.
   * - **Debug grid**
     - Show the 2×2 diagnostic view at start.

Three resolutions are independent and it is worth keeping them straight:
camera capture resolution controls what comes in, background resolution
controls the cached and final output size, and calculation scale controls the
internal mask and FFT geometry. The quality presets set the last of these for
you; **Custom** reveals it directly.

On macOS, colour-key thresholding and lens-map construction are offloaded to
the GPU through Metal. The rest of the pipeline is multi-threaded with OpenMP
and threaded FFTW3 plans, and each stage accepts only the most recent frame, so
a slow stage drops frames instead of building a backlog.

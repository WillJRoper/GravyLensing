Command line arguments
======================

Every flag is optional. Flags seed the startup dialog rather than bypassing it;
anything omitted falls back to the last saved session setting. Boolean flags
have an explicit ``--no-`` counterpart, and passing both forms of the same flag
is an error.

.. code-block:: text

   Usage: ./gravy_lens [options]

   Options:
     -n, --nthreads <n>               Lensing worker threads (default: cores minus 2).
     -s, --strength <f>               Lens strength multiplier (default 4.0).
     -f, --softening <f>              Kernel softening radius in px (default 50.0).
     -m, --visionSize <n>             Vision request size (default 512).
     -d, --deviceIndex <n>            Camera device index (default 0).
     --fps, --frameRate <n>           Target camera frame rate (default 30).
     -g, --debugGrid                  Show 2x2 diagnostic grid at start.
     --no-debugGrid                   Force the debug grid off.
     -p, --padFactor <n>              FFT padding multiplier (default 2).
     -t, --temporalSmooth <f>         Mask temporal blending factor (default 0.25).
     --personSensitivity <n>          Person sensitivity, 0-100 (default 50).
     --lr, --lowerRes <f>             Internal calculation scale, 0.1-1.0.
     --quality, --qualityMode <mode>  fast, balanced, high, or custom.
     --sb, --secondsPerBackground <n> Seconds per background; -1 = manual (default -1).
     --di, --distortInside            Also lens the interior of the mask (default on).
     --no-distortInside               Force interior distortion off.
     --flip                           Mirror camera feed horizontally.
     --no-flip                        Force mirroring off.
     --roi, --selectROI               Open ROI selector on first session start.
     --no-selectROI                   Skip startup ROI selector.

The flags are most useful for scripting a known-good configuration for a
particular room, so that a machine set up the night before comes back in the
same state:

.. code-block:: bash

   # Mirror mode, ROI prompt at startup, manual background switching
   ./gravy_lens --flip --roi --sb -1

.. code-block:: bash

   # Telescope mode, no mirroring, backgrounds cycling every 20 seconds
   ./gravy_lens --no-flip --sb 20

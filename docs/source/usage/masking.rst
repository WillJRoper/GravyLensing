Lens modes and masking
======================

The mask is the heart of the demo: it is the projected "mass" that does the
lensing. Everything else in the pipeline is downstream of it, so a noisy or
flickering mask shows up as a noisy, flickering lens. There are two ways to
build one, switchable live with :kbd:`Shift+M` or from the **Lens** menu.

People
------

Person detection uses Apple's native Vision person segmentation. There is no
model to download and no training step; it runs on the machine and it is the
default for good reason — it works out of the box, on anyone, in most lighting.

The controls live under **Person Detection** in settings:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Control
     - What it does
   * - **Quality mode**
     - Fast, Balanced, High Quality, or Custom. Picks sensible values for
       detection detail and calculation scale. Balanced is the recommended
       starting point; Custom exposes the underlying controls.
   * - **Detection detail**
     - The size, in pixels, of the frame handed to Vision. Larger is more
       accurate and slower.
   * - **Mask stability**
     - How strongly each new mask is blended with the previous one. Higher
       values are steadier but lag behind fast movement.
   * - **Sensitivity**
     - 0 is strict, 100 is permissive. Lower it to drop faint or distant
       figures; raise it if the subject keeps dropping out.
   * - **Focus mode**
     - "Track main subject only" — keeps only the largest connected group of
       people and ignores bystanders elsewhere in frame.
   * - **Group distance**
     - How close two people must be, as a fraction of the frame width, to
       count as the same group. Raise it to keep a small group together.

Focus mode is the main tool for the crowded-room problem; see
:doc:`../running_modes` for how it fits with the physical setup.

Selected colour
---------------

Colour mode lenses on a colour rather than a person. Press :kbd:`Shift+S` (or
**Lens > Select Colour...**) and click the target object in the picker window
that opens; :kbd:`Esc` or :kbd:`c` cancels. The measured spread of the sampled
pixels becomes the starting tolerance, so click the middle of the object rather
than its edge or a highlight.

There are two sub-modes:

**Match every pixel** (recommended)
   A chroma-key style matte: every pixel within tolerance of the target colour
   becomes mass. Fast, stable, and the most predictable. Best for coloured
   areas, clothing, and green-screen style effects.

**Track one coloured object** (advanced)
   Follows a single connected region of that colour across frames, using blob
   continuity to stay on it. Use this when other parts of the scene contain
   similar colours and the flat matte picks up too much.

The tolerance controls are stated in plain terms in the dialog:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Control
     - What it does
   * - **Colour range**
     - How far the hue may drift from the target. Increase when parts of the
       object are missed; decrease when unrelated colours creep in.
   * - **Vividness range**
     - Saturation tolerance. Widen it when the object has shadowed and lit
       faces.
   * - **Brightness range**
     - Value tolerance. The first thing to widen when the object moves between
       bright and dim parts of the room.
   * - **Minimum object size**
     - Regions smaller than this are discarded — the main defence against
       speckle.
   * - **Tracking persistence**
     - How many frames a lost object is remembered before the mask gives up.
       Raises tolerance to brief occlusion.
   * - **Mask stability**
     - Temporal smoothing of the final mask, as in person mode.

Colour mode is the reliable way to escape the background-people problem: give
the subject a vivid, saturated, matte object — a ball works well — in a colour
nothing else in the room shares. Avoid pastels, avoid white and black, and
avoid anything shiny, since a specular highlight reads as a different colour
entirely.

Region of interest
------------------

Both lens modes can be restricted to a rectangle of the frame. Press
:kbd:`Shift+R` (or **Lens > Select Region...**), drag out the region, and only
that part of the frame is searched for mass. **Lens > Use Full Frame** clears
it.

An ROI around a marked standing spot is the single most effective way to stop
the demo reacting to passers-by, and it costs nothing in performance — it
reduces the work done rather than adding to it. The selection survives session
restarts.

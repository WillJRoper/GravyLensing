Interface and shortcuts
=======================

Sessions
--------

All configuration goes through an explicit session. The **Session Setup**
dialog opens at launch with your previous choices remembered; changing anything
mid-session means reopening settings and restarting.

1. Open **Session > Settings...** (:kbd:`Cmd+,`).
2. Edit the configuration.
3. Click **Restart Session**.

The current colour target and region of interest are preserved across restarts
where possible. Live mode and debug-grid toggles also survive.

The settings dialog is scrollable and works on small laptop displays. It shows
only the controls relevant to the selected lens mode, hides technical controls
behind the **Custom** presets, and validates camera and background choices in
place. **Restore Defaults** resets every control to the shipped defaults.

Camera
------

**Capture resolution** can be Automatic, 480p, 720p, or 1080p; the selected
camera's actual negotiated format appears in the session window title, which is
worth checking when a camera quietly ignores a request. **Frame rate** and
**Flip** (horizontal mirroring) are set here too — see :doc:`../running_modes`
for which way mirroring should go.

The debug grid
--------------

:kbd:`Shift+D` toggles a 2×2 diagnostic view: camera feed, mask, deflection,
and final output. When something looks wrong, this tells you which stage is at
fault — a bad mask and a bad lens look identical in the final image, and
completely different here. Learn to read it before you need it.

Keyboard shortcuts
------------------

.. list-table::
   :header-rows: 1
   :widths: 44 22 34

   * - Action
     - Shortcut
     - Menu
   * - Switch lens mode
     - :kbd:`Shift+M`
     - Lens > Switch Lens Mode
   * - Select / reselect colour
     - :kbd:`Shift+S`
     - Lens > Select Colour...
   * - Select region of interest
     - :kbd:`Shift+R`
     - Lens > Select Region...
   * - Clear the region
     - —
     - Lens > Use Full Frame
   * - Previous / next background
     - :kbd:`Left` / :kbd:`Right`
     - Background > Previous / Next
   * - Toggle debug grid
     - :kbd:`Shift+D`
     - View > Debug Grid
   * - Show camera inside the lens
     - :kbd:`Shift+I`
     - View > Show Camera Inside Lens
   * - Toggle full screen
     - :kbd:`Ctrl+Cmd+F`
     - View > Toggle Full Screen
   * - Session settings
     - :kbd:`Cmd+,`
     - Session > Settings...
   * - Quit
     - :kbd:`Cmd+Q`
     - File > Quit

**Help > Keyboard Shortcuts** shows the same list inside the app, which is the
version to trust if the two ever disagree.

Quick start
===========

Launch the app from Applications, or run the built binary:

.. code-block:: bash

   ./gravy_lens

The **Session Setup** dialog opens first, with your choices from last time
already filled in: what to detect, which camera, quality, region, and
backgrounds. The defaults are deliberately safe — if you have no reason to
change anything, click **Start Session**.

Your first session
------------------

1. Leave the lens mode on **People**.
2. Click **Start Session**.
3. Stand in front of the camera. The background image bends around you.
4. Press :kbd:`Left` and :kbd:`Right` to change background image.
5. Press :kbd:`Shift+D` to show the 2×2 diagnostic grid — camera, mask,
   deflection, and output. This is the fastest way to see *why* the demo is
   misbehaving.
6. Press :kbd:`Cmd+,` to reopen settings, edit, and click **Restart Session**.

Trying colour mode
------------------

1. Press :kbd:`Shift+M` to switch to **Selected Colour**, or choose
   **Lens > Selected Colour**.
2. Press :kbd:`Shift+S` and click the object you want to lens with in the
   picker window that opens. :kbd:`Esc` cancels.
3. Move the object around. The lensing follows the colour, not the person.

A brightly coloured ball works far better than a pastel jumper. See
:doc:`../usage/masking` for why, and what to do when the mask is noisy.

Where to go next
----------------

Before running this for an audience, decide how the room is laid out:
:doc:`../running_modes` covers the two ways the demo is normally set up, and
:doc:`../recommendations` covers making it robust once people are watching.

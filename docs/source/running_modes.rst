Ways to run the demo
====================

GravyLensing can be set up in two physically different ways. They use the same
software and the same settings, but they put the camera, the screen and the
subject in different places, and they fail in different ways. Choosing between
them is the first decision to make when planning a demo, because everything
else — room layout, lighting, which lens mode to use — follows from it.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Mirror mode
     - Telescope mode
   * - Camera points at
     - The audience
     - The screen
   * - Subject stands
     - In front of the screen, facing it
     - Between the camera and the screen
   * - Mirroring
     - On, so movement matches
     - Off
   * - Main weakness
     - Bystanders in the background get lensed too
     - Needs space, and careful field-of-view setup
   * - Best for
     - Drop-in crowds, small spaces
     - A controlled space, a faithful analogy

Mirror mode
-----------

The camera sits next to or above the screen and looks back at the person, who
watches themselves on the screen. The feed is mirrored horizontally so that
moving left on the screen matches moving left in the room, exactly like a
mirror. This is the intuitive setup: people work out what they are looking at
in a second or two, and start playing with it immediately.

Turn mirroring on with **Mirror camera feed** in settings, or ``--flip`` on the
command line.

The catch is that *everything the camera sees* is a candidate mass. Anyone
walking past behind the subject is detected as well, and the background image
warps around them too. In a busy room the effect stops reading as "that person
is bending spacetime" and starts reading as visual noise.

There are several options to push back on this, and they stack:

- **Focus mode** (**Track main subject only**) keeps only the largest connected
  group of people in the mask and ignores bystanders elsewhere in the frame.
  **Group distance** controls how close two people must be, as a fraction of
  the frame width, to count as one group — raise it to keep a small group
  together, lower it to isolate one individual.
- **A region of interest** (:kbd:`Shift+R`) restricts detection to a rectangle
  of the frame. If the subject stands on a marked spot, an ROI around that spot
  removes the rest of the room from the problem entirely.
- **Colour mode** sidesteps people altogether: hand the subject a brightly
  coloured object and lens on that. Bystanders are then irrelevant unless they
  happen to be wearing the same colour.
- **Person sensitivity** can be lowered so that faint, distant or partially
  occluded figures fall below the detection threshold.

These help, but none of them beats physically isolating the subject. A backdrop
behind the subject, a marked standing spot, a barrier, or simply pointing the
camera slightly downward so that it sees floor rather than the queue, will do
more for the demo than any setting. Treat the software options as a safety net
for the people who wander in anyway, not as the plan.

Telescope mode
--------------

The camera points *at the screen*, and the subject stands in between, in the
camera's line of sight. What the camera sees is the screen — the distant source
— with a person silhouetted against it. That silhouette becomes the lensing
mass, and the screen shows the source deflected around them.

This is the faithful version of the analogy. It is the geometry of a real
observation: a distant source, an intervening mass, and a telescope looking
through the mass at the source. The audience is standing inside the light path,
which is a much better story to tell than "the computer is drawing a warp
around you".

Mirroring should be **off** in this mode. The camera and the screen already
face each other, so the image is not being used as a mirror and flipping it
will make movement run the wrong way.

Telescope mode is far less prone to the background problem: the backdrop behind
the subject is the screen itself, and anyone who is not between the camera and
the screen is simply not in shot. In exchange it demands more from the space:

- **The camera must be far enough back.** Too close and the screen does not
  fill the frame, so the illusion breaks at the edges where the camera sees the
  wall, the screen bezel, or the room beyond. Distance is what makes it feel
  seamless.
- **The field of view has to be matched to the screen.** Aim for the screen to
  fill the frame as completely as you can manage, and use a region of interest
  (:kbd:`Shift+R`) to trim whatever is left over around the edges.
- **The subject needs room to move without leaving the shot** — and without
  standing so close to the camera that they fill the whole frame.
- **Lighting is a trade-off.** A bright room lights the subject nicely but
  washes out the screen; a dark room does the reverse. The subject only has to
  be distinguishable from the screen behind them, not well lit.
- **Person detection is usually the safer lens mode here**, because the
  backdrop is a moving, high-contrast image. Colour keying against a changing
  screen is much harder than colour keying against a wall.

Which one should you use?
-------------------------

If you have a small space, a drop-in audience, and no control over who walks
through, use mirror mode with focus mode enabled and an ROI around the
subject's spot. It is more forgiving to set up and it explains itself.

If you have a dedicated space you can lay out in advance, use telescope mode.
It is a better demonstration of the physics, and it solves the crowd problem by
construction rather than by settings.

Setting either up in a real room is covered in :doc:`recommendations`.

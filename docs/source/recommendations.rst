Running a good demo
===================

The software works out of the box. Making it work *in a room, in front of
people, for six hours* is a different problem, and it is mostly a physical one.
This page collects what to do about that. Read :doc:`running_modes` first — the
advice here assumes you have chosen one.

Set the room up first, the settings second
------------------------------------------

Almost every problem the demo has in the field is a problem with the space, and
almost every setting exists to paper over one. Papering works, but it costs
robustness, and a run that depends on five compensating settings is a run that
breaks when the light changes at four o'clock.

In rough order of how much they matter:

1. **Isolate the subject.** A marked spot on the floor, a backdrop, a barrier,
   or a camera angled down at the floor rather than across the room. This one
   change removes more failure modes than every software option combined.
2. **Control what is behind them.** In mirror mode, whatever is behind the
   subject is competing for the mask. In telescope mode, the screen is the
   backdrop and this is solved by construction.
3. **Get the lighting steady.** Even, constant light beats bright light.
   Daylight through a window will change over the day and take your colour
   calibration with it.
4. **Then tune the settings**, with the room as it will actually be.

A pre-run checklist
-------------------

- Start the session and check the window title reports the camera format you
  expected — cameras quietly ignore resolution requests.
- Turn on the debug grid (:kbd:`Shift+D`) and watch the mask, not the output,
  while someone walks around the space. Everything you are about to tune is
  visible there.
- Walk the edges of the area: does the subject stay masked at the extremes of
  where they will stand?
- If using colour mode, re-pick the target colour *in the room's real
  lighting*, with the object held where it will actually be held.
- Check the frame rate under load, with the subject moving, not standing still.
- Set an ROI around the area the subject will occupy.
- If it will run unattended, enable auto-cycling so the background changes
  without anyone touching the machine.

Tuning order when the mask misbehaves
-------------------------------------

Work down this list rather than turning several knobs at once:

**The mask flickers or breaks up**
   Raise **Mask stability** first. Then, in colour mode, raise **Minimum
   object size** to kill speckle and **Tracking persistence** to survive brief
   dropouts.

**Bystanders are being lensed**
   Enable **Focus mode**, then set a **region of interest**, then lower
   **Sensitivity**. If it is still a problem, the answer is a backdrop or
   telescope mode, not another setting.

**The subject drops out**
   Raise **Sensitivity**, raise **Detection detail**, and check the lighting on
   the subject. In colour mode, widen **Brightness range** before anything else
   — it is nearly always a lighting change, not a hue change.

**The edges look hard or jagged**
   Raise **Lens edge softness**. This is independent of detection quality, so
   it is cheap.

**It is too slow**
   Lower **Processing size**, then **Calculation scale**, then the quality
   preset. Lower resolution before frame rate: a smaller image at 30 fps reads
   much better than a large one at 12.

Making the physics land
-----------------------

- **Pick backgrounds with structure.** A grid or a field of galaxies makes the
  deflection obvious; a smooth gradient lenses just as correctly and looks like
  nothing is happening.
- **Do not max out the strength.** Past a point the background stops looking
  deflected and starts looking smeared, and the effect reads as a filter rather
  than as physics. Moderate strength with a well-chosen background is more
  convincing.
- **Let people move slowly.** An Einstein ring appearing as someone lines up
  with a bright feature is the moment worth waiting for, and it is missed
  entirely at a run.
- **The debug grid is a teaching tool.** Showing the mask next to the output
  makes it obvious that the app is measuring a mass distribution and computing
  a deflection, rather than drawing a blob.

Unattended and long runs
------------------------

- Disable screen sleep and system sleep on the machine.
- Turn on background auto-cycling so the display keeps changing.
- Full screen (:kbd:`Ctrl+Cmd+F`) hides the menu bar and the temptation to
  click things.
- Prefer conservative settings over the best-looking ones. The configuration
  that survives an unexpected crowd, a light change and a flat battery is
  better than the one that looks marginally sharper at ten in the morning.
- Seed the configuration from the command line (see :doc:`usage/cli`) so a
  machine that has been restarted comes back in a known state.

FAQ
===

A collection of questions that come up when running GravyLensing. It is meant
to grow — if you hit something that is not here, please open an issue or add a
section.

Everyone walking past gets lensed. How do I stop that?
------------------------------------------------------

This is the most common problem with mirror mode, where the camera faces the
room and every person in shot is a candidate mass. In order of effectiveness:

1. Isolate the subject physically — a backdrop, a marked spot, or a camera
   angled down at the floor rather than across the room.
2. Enable **Focus mode** ("Track main subject only"), which keeps only the
   largest connected group of people in the mask.
3. Set a **region of interest** (:kbd:`Shift+R`) around where the subject
   stands, so the rest of the frame is never searched.
4. Switch to colour mode and lens on a vivid object the subject holds.
5. Lower **Sensitivity** so faint and distant figures fall below threshold.

Or use telescope mode, where the problem largely does not arise. See
:doc:`running_modes`.

Should the image be mirrored or not?
------------------------------------

Mirror it when the camera faces the subject and they are watching themselves —
movement then matches, like a mirror. Do not mirror it in telescope mode, where
the camera and screen face each other; flipping there makes movement run the
wrong way.

How far back does the camera need to be in telescope mode?
----------------------------------------------------------

Far enough that the screen fills the frame. There is no single number, because
it depends on the screen size and the camera's field of view. Set it up with
the debug grid on and back the camera off until the screen fills the camera
view with a little to spare, then trim the remainder with a region of interest.
Too close is the common mistake: the illusion breaks wherever the camera sees
past the edge of the screen.

The mask flickers, and the lens flickers with it
------------------------------------------------

Raise **Mask stability**, which blends each mask with the previous one. In
colour mode also raise **Minimum object size** to remove speckle and
**Tracking persistence** so a brief dropout does not clear the mask. If it is
still unstable, the detection itself is marginal — improve the lighting or the
colour contrast rather than smoothing harder, since heavy smoothing shows up as
lag.

Colour tracking worked this morning and not this afternoon
----------------------------------------------------------

The light changed. Re-pick the target colour (:kbd:`Shift+S`) in the current
lighting, and widen **Brightness range** first — daylight shifts brightness far
more than hue. For a long run, prefer light you control over daylight.

Which colour object works best?
-------------------------------

Vivid, saturated, matte, and a colour nothing else in the room shares. A
coloured ball is close to ideal. Avoid pastels and anything near white or
black, since there is little hue to lock onto, and avoid shiny surfaces —
a specular highlight reads as a completely different colour and punches a hole
in the mask.

Nothing looks like it is being lensed
-------------------------------------

Check the background image first. Smooth, near-uniform images are deflected
just as correctly as detailed ones, but the result looks identical to the
original. Use a background with structure — a grid or a field of galaxies —
before reaching for the strength control.

Should I just turn the strength up?
-----------------------------------

Usually not. Past a moderate setting the background stops reading as deflected
and starts reading as smeared, which makes the demo look like a filter rather
than physics. A well-chosen background at moderate strength is more convincing
than a maxed-out one.

The frame rate is poor
----------------------

Lower **Processing size** first, then **Calculation scale**, then drop the
quality preset. Lower the resolution before the frame rate: a smaller image at
30 fps reads much better than a large one at 12. Background resolution drives
FFT memory use directly, which is why it is the first lever.

The edges of the frame look wrong, as if the distortion wraps around
--------------------------------------------------------------------

That is FFT wrap-around. Raise **Edge protection** (the padding factor) from
Standard to Extra or Maximum.

Why does the first start take longer after I change background settings?
------------------------------------------------------------------------

Backgrounds are resized and cached at the selected output size before a session
starts. Changing the directory or the processing size invalidates that cache
and it is rebuilt once. The Backgrounds page reports the cache size and can
rebuild it on demand, which is the fix if cached images ever look stale.

Can I use my own background images?
-----------------------------------

Yes. Open **Session > Settings...**, point the backgrounds source at your own
directory, and restart the session. Supported images are discovered
automatically. **Restore Defaults** returns to the packaged set.

The camera is not the one I wanted, or the resolution is not what I asked for
-----------------------------------------------------------------------------

Pick the camera by name in settings rather than by index where possible. The
format the camera actually negotiated is shown in the session window title —
cameras routinely ignore a resolution request and give you something else, and
the title is how you find out.

Does it run on anything other than macOS?
-----------------------------------------

The pipeline is portable C++ with Qt6, OpenCV and FFTW3, but person
segmentation uses Apple Vision and the GPU path uses Metal, so macOS is the
supported platform. The released DMG requires Apple Silicon and macOS 14 or
newer.

Do I need to be online?
-----------------------

No. Everything runs locally, with no model download and no network access at
any point.

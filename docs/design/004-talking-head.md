# 004 — Talking head

## Problem
The companion can hear, think, and speak, but has no face — replies are audio only. The next step is a 2D talking head: a still image the user uploads, with the mouth animated in sync with `Speaker`'s TTS output, running on this machine (no NVIDIA GPU — Intel UHD 620 integrated graphics only, confirmed via `Get-CimInstance Win32_VideoController`).

## Goals
- Animate the mouth of a single user-supplied 2D image in sync with spoken replies, in real time, on CPU only.
- No hand-drawn mouth-shape assets required — one photo/artwork in, talking head out.
- Keep the same `say(text)` / `interrupt()` / `stop()` interface `Speaker` already has, so `main.py` barely changes and barge-in keeps working.

## Non-goals
- Photorealism. This is a cheap, visibly-synthetic effect, not a substitute for a rendered video. (Idle motion — blinking, a bob, an occasional head twitch — was added later; see "Idle motion" below.)
- Full viseme accuracy — Rhubarb's 9-shape set mapped onto a geometric warp of a still photo will never match a trained lip-sync model. That tradeoff is intentional (see Alternatives).
- Real-time 3D or GAN-based reenactment (SadTalker, MuseTalk, Wav2Lip, LivePortrait, etc.) — all require an NVIDIA GPU to run at usable speed; this machine doesn't have one.

## Chosen approach
Two well-known, actively maintained, CPU-friendly open-source pieces, glued together with a small amount of custom compositing code:

1. **[Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync)** (`rhubarb.exe`, external binary, not a pip package) analyzes a WAV recording and emits timestamped mouth-shape cues from its 9-shape set (A–H, X — see its README). It's fast (sub-second for a short phrase) and Windows binaries are published on its GitHub releases page.
2. **[MediaPipe Face Mesh](https://github.com/google-ai-edge/mediapipe)** (`mediapipe`, pip package) runs once per uploaded image, in static-image mode, to locate the lip landmarks (`FACEMESH_LIPS`) on that specific photo/artwork.

At startup, `ai_girlfriend.avatar.face` uses those landmarks to precompute one warped frame per Rhubarb viseme (9 frames total) by displacing pixels in a mouth-centered region with a smooth, Gaussian-falloff field — vertical displacement for open/closed shapes, horizontal for wide/rounded ones — so there's no seam and no per-frame warp cost during playback. Precomputing means real-time playback is just swapping between 9 cached images on a schedule, which is trivial on CPU regardless of warp complexity.

`ai_girlfriend.avatar.player.Avatar` replaces `Speaker` as the thing `main.py` calls `say()` on:
1. Synthesize the reply text to a temp WAV file via SAPI5's `SpFileStream` (same COM object family `Speaker` already uses, just rendering to a file instead of speaking live).
2. Run `rhubarb.exe` on that WAV (with `--dialogFile` set to the reply text, per Rhubarb's own recommendation for more reliable results) to get viseme cues.
3. Play the WAV via `winsound.PlaySound` (stdlib, Windows-only — already an implicit platform constraint via SAPI5) and, on the same worker thread, swap the displayed frame in a small `pygame` window according to the cue timeline, using one `time.monotonic()` clock for both — so there's no separate synthesis pass to drift out of sync with what's audibly played.

This mirrors `Speaker`'s worker-thread/queue design exactly, including `interrupt()` (stop audio, drop queue, reset to the idle "X" frame) and `stop()` (same, plus shut down the worker and close the window).

## Interface contract
```python
from ai_girlfriend.avatar.player import Avatar

avatar = Avatar(image_path="my_photo.png", rhubarb_path="rhubarb.exe")
avatar.say("Hello there!")   # non-blocking, mirrors Speaker.say()
avatar.interrupt()           # cuts off current playback, drops anything queued
avatar.stop()                # shuts down worker + closes window
```
`main.py` picks `Avatar` instead of `Speaker` when `AVATAR_ENABLED=true` and both are otherwise interchangeable from its point of view. `interrupt()` no longer doubles as live barge-in — see "Microphone self-feedback" below for why.

## Alternatives considered
- **Wav2Lip / SadTalker / MuseTalk / LivePortrait** (GAN or diffusion-based video generation from a driving photo + audio): the standard, much higher-fidelity approach — but all assume an NVIDIA GPU. On this CPU-only/Intel-iGPU machine they'd either fail to run or take tens of seconds to minutes per reply, breaking the live-conversation feel every other module in this project is built around. Worth revisiting if a GPU becomes available (mirrors the same call already made for STT device and the local-LLM alternative in 003).
- **PNGTuber-style amplitude-only mouth swap** (2-3 states — open/closed — driven by audio volume, no phoneme analysis): simpler still, but noticeably less convincing since it ignores which sound is being made. Rhubarb's per-phoneme cues are nearly as cheap to compute and look substantially better, so there was no real reason to settle for less.
- **Requiring the user to hand-draw mouth-shape variants**: gives crisper, more stylized results with less warping artifacts, but adds real friction (the user explicitly wants to start from a single image). Revisit as an optional override later — `face.py`'s precomputed-frame-per-viseme design would accept hand-drawn frames as a drop-in replacement for the auto-warped ones without changing `player.py` at all.

## Idle motion
Added after the initial mouth-only version, once the auto-warped look was validated by eye: she now never sits perfectly frozen, whether idle or mid-reply.

- **Segmentation compositing.** `face.py` runs MediaPipe's Image Segmenter (`selfie_segmenter.tflite`) once per image to separate her from the background, then `cv2.inpaint`s a static background plate to fill in what's behind her (`build_background`). `player.py` draws that plate once per frame and composites her BGRA cutout on top, so the animations below move only her, not the whole photo. (`output_confidence_masks=True` segfaults with this mediapipe/model combination — `output_category_mask=True` is used instead, with the resulting hard edge softened by a small Gaussian blur.)
- **Blink** (`blink.py`): an asymmetric two-eye state machine — one eye closes, then the other a beat later, then both reopen the same way — on a randomized interval, independent of speech.
- **Idle bob + squash-and-stretch** (`bob.py`, `squash.py`): a continuous vertical "V" bob, eased with a smootherstep curve so she holds near the top/bottom and snaps quickly through the middle rather than moving at a constant rate. The bob's phase directly drives a tied squash-and-stretch scale, anchored at the bottom edge (not the center) so she reads as standing on a fixed floor and growing/shrinking from there, not swelling from her own middle.
- **Head twitch** (`twitch.py`): a rare (every 25-75s), quick back-and-forth rotation snap, decaying over a few steps.

## Microphone self-feedback
There's no acoustic echo cancellation in this pipeline. Without it, her own voice playing through the speakers leaks into the microphone and gets misread by the VAD as the user talking — which both self-interrupts her mid-reply and occasionally gets transcribed and answered as if the user had said it themselves. Fixed by having `Speaker`/`Avatar` accept `on_playback_start`/`on_playback_end` callbacks, fired around exactly the audible span of their own output (not synthesis, which SAPI5 renders silently to a file first); `main.py` wires these to `Listener.mute()`/`unmute()`, which toggle RealtimeSTT's `set_microphone()`. The tradeoff: real barge-in (the user actually talking over her) no longer works while she's speaking, since the mic is off for that window — the standard way small voice-assistant setups without a headset/AEC handle this problem.

## Future work
- Let the user supply their own per-viseme images to replace the auto-warp.
- Real acoustic echo cancellation (e.g. via a headset, or a proper AEC library) to restore live barge-in without the self-feedback problem.
- Reassess against Wav2Lip/SadTalker if a GPU becomes available later.

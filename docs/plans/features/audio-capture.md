# Audio Capture for Animal Detection Videos

## Overview

Record audio via a USB microphone in parallel with every video capture. After analysis, if an animal is detected, the audio is muxed into the video using FFmpeg. Otherwise, the temporary audio file is deleted.

## Flow

1. Motion/schedule triggers a video capture
2. `AudioRecorder` starts recording concurrently with the camera
3. Video (`.mp4`) and audio (`.wav`) are saved to disk
4. `_process_capture` runs classification + detection
5. If an animal is detected → FFmpeg muxes audio into the video, deletes the `.wav`
6. If no animal → deletes the `.wav`

## Implementation Steps

### 1. New module: `src/capture/audio_recorder.py`

- Uses **FFmpeg subprocess** to record from the USB mic — no new dependencies since FFmpeg is already required by `picamera2`'s `FfmpegOutput`.
- `start(output_path, duration)` — spawns `ffmpeg -f alsa -i plughw:1,0 -t {duration} -y output.wav` in a background process.
- `stop()` / `wait()` — terminates or waits for the process to finish.
- Graceful fallback: if no mic is found or FFmpeg fails, logs a warning and returns `None`. Recording should never block or break video capture.

### 2. Modify `Camera.capture_video()` in `src/capture/camera.py`

- Accept an optional `AudioRecorder` instance (injected at `__init__`).
- In `_record_video()`, call `audio_recorder.start()` just before `start_encoder()` and `audio_recorder.wait()` after `stop_encoder()`, so audio and video are captured concurrently for the same duration.
- Add `audio_path: Optional[Path]` field to the `VideoMetadata` dataclass.
- Simulation mode: skip audio recording.

### 3. New utility: `src/capture/audio_mux.py`

- `mux_audio(video_path, audio_path) -> bool` — runs `ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -movflags +faststart output.mp4`, then atomically replaces the original video and deletes the WAV.
- `-c:v copy` avoids re-encoding video.
- `-c:a aac` ensures browser-compatible audio.
- `-movflags +faststart` enables streaming playback in the UI.

### 4. Modify `_process_capture()` in `main.py`

- After classification + detection, if an animal **is** detected (`result.is_animal`) and `metadata.audio_path` exists → call `mux_audio()`.
- If no animal detected → delete the `.wav` file to save disk space.

### 5. Configuration in `config.yaml`

```yaml
audio:
  enabled: true
  device: "plughw:1,0"   # ALSA device for USB mic (use `arecord -l` to find)
  sample_rate: 44100
```

### 6. Wiring in `main.py` / `WildlifeMonitor.__init__`

- Read `audio` config, instantiate `AudioRecorder`, and pass it to `Camera`.
- Feature is fully opt-in: if `audio.enabled` is `false` or the mic is absent, everything works as before.

## What stays unchanged

- **Web UI video playback** — `<video>` already supports MP4 with AAC audio, no changes needed.
- **Timelapse generation** — continues to be video-only (no audio to stitch).
- **Database schema** — no new columns required; audio lives inside the existing `.mp4` file.

## Dependencies

No new pip dependencies. FFmpeg (already required) handles both recording and muxing. The only prerequisite is `alsa-utils` on the Pi (`sudo apt install alsa-utils`), which is pre-installed on Raspberry Pi OS.

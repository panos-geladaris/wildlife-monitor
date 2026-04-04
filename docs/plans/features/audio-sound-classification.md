# Audio Sound Classification

## Overview

Classify animal sounds from audio captured alongside video. When a video is captured, the recorded audio is analysed to identify whether it contains an animal sound and, if so, what species or category it belongs to. Results are stored in the database and displayed on the detection and gallery pages.

## Prerequisites

- Audio capture feature is enabled and working (see `audio-capture.md`)
- `.wav` files are already recorded alongside each video via `AudioRecorder`
- `panns-inference` and `birdnet` installed (both listed in `requirements.txt`)

## Two-Pass Classification Strategy

Audio analysis uses two complementary models in sequence:

```
Audio file (.wav)
  │
  ├─► Pass 1: PANNs (broad) ──► "Is there an animal sound?" + category
  │                               (dog, cat, frog, insect, bird, ...)
  │
  └─► Pass 2: BirdNET (specific) ─► Species-level bird ID
                                     (runs when Pass 1 or visual classifier
                                      indicates "bird", OR when PANNs is
                                      not installed)
```

### Pass 1 — PANNs (general animal sound detection)

- **Model**: CNN14 from [panns_inference](https://github.com/qiuqiangkong/panns_inference) (PyTorch-based)
- **Purpose**: Broad animal sound detection across all categories
- **Output**: AudioSet class label + confidence (e.g. "Dog — 0.82")
- **AudioSet animal classes**: dog, cat, frog, insect, bird, livestock, rodent, snake, wild_mammal, animal (fallback)
- **Behaviour**: If no animal sound is found above `panns_min_confidence` → `sound_class = NULL`, analysis stops. The visual detection stands on its own.
- **Generic "Animal" label**: Treated as a fallback — used only when no more specific category exceeds the threshold.

### Pass 2 — BirdNET (bird species identification)

- **Model**: BirdNET v2.4 via the [`birdnet`](https://github.com/birdnet-team/birdnet) Python package (TFLite backend)
- **Purpose**: Species-level identification for bird sounds
- **Trigger**: Runs when:
  - Pass 1 detected a bird-category sound, **or**
  - The visual classifier detected `animal_class = "bird"`, **or**
  - PANNs is not installed (BirdNET acts as a standalone pass-1 detector)
- **Output**: Species name + confidence (e.g. "Turdus merula_Eurasian Blackbird — 0.91")
- **Location filtering**: Supports lat/lng filters — reuses `daylight.lat` / `daylight.lng` config values

## Configuration

```yaml
# Audio sound classification (requires audio capture enabled)
audio_analysis:
  enabled: false
  panns_enabled: true              # Pass 1: broad animal sound detection
  panns_min_confidence: 0.3        # PANNs minimum confidence threshold
  birdnet_enabled: true            # Pass 2: bird species identification
  birdnet_min_confidence: 0.5      # BirdNET minimum confidence threshold
  # Location-based filtering for BirdNET (reuses daylight lat/lng if not set)
```

Both `panns_enabled` and `birdnet_enabled` can be toggled independently. Each model has its own confidence threshold: PANNs defaults to `0.3` (broader recall), BirdNET defaults to `0.5` (higher precision for species ID).

Note: `audio.enabled` must also be `true` for audio to be captured in the first place.

## Implementation

### `src/analysis/audio_classifier.py`

```python
@dataclass
class AudioClassificationResult:
    sound_class: Optional[str]        # Category: "bird", "dog", "frog", "insect", None
    sound_species: Optional[str]      # Species (birds only): "Turdus merula_Eurasian Blackbird"
    sound_confidence: Optional[float] # Confidence of the best match
    all_predictions: list[tuple[str, float]]  # Raw PANNs predictions for debugging

class AudioClassifier:
    def __init__(
        self,
        panns_min_confidence: float = 0.3,
        birdnet_min_confidence: float = 0.5,
        panns_enabled: bool = True,
        birdnet_enabled: bool = True,
        lat: float = None,
        lng: float = None,
    ): ...

    def classify_audio(
        self,
        wav_path: Path,
        visual_animal_class: Optional[str] = None,
    ) -> AudioClassificationResult: ...
```

Key design points:
- Lazy model loading — models loaded on first call, not at init
- Graceful degradation: if `panns_inference` not installed or `panns_enabled=False` → skip Pass 1; same for BirdNET
- When PANNs is unavailable, BirdNET runs unconditionally on every file (acts as standalone detector)
- Generic "Animal" AudioSet label used only as a fallback when no specific category exceeds threshold

### Database schema

Three nullable columns on the `detections` table:

```sql
ALTER TABLE detections ADD COLUMN sound_class TEXT;
ALTER TABLE detections ADD COLUMN sound_species TEXT;
ALTER TABLE detections ADD COLUMN sound_confidence REAL;
```

Auto-migrated in `_init_db()` if columns are absent.

### Main pipeline (`main.py`)

Audio classification runs in `_process_capture()` after visual classification and object detection:

1. Check `self._audio_classifier` is initialised and `metadata.audio_path` exists
2. Pass `visual_animal_class` to help decide BirdNET trigger
3. If `audio_result.has_sound` → update `sound_class`, `sound_species`, `sound_confidence` in DB

Audio file retention: the WAV is kept (muxed into the MP4) if either a visual animal **or** a sound was detected. Previously, only the visual result was checked — this meant audio-only detections had their WAV discarded immediately after classification.

### Reclassification (`POST /api/detections/<id>/reclassify`)

Re-classification also re-runs audio analysis if the original `.wav` file is still present alongside the video (same stem, `.wav` extension). The step is silently skipped if no WAV exists (the common case — it is muxed into the MP4 or discarded at capture time). Sound fields are included in the reclassify response.

### Cleanup (`src/storage/cleanup.py`)

`cleanup_detection()` now also deletes `<video_stem>.wav` if it exists alongside the video. Orphaned WAVs can accumulate when the process crashes between audio capture and the mux/discard step at the end of `_process_capture()`.

### API

Sound fields are included in all detection serialisation responses:

```python
"sound_class": d.sound_class,
"sound_species": d.sound_species,
"sound_confidence": d.sound_confidence,
```

Affected endpoints: `GET /api/detections`, `GET /api/detections/<id>`, `GET /api/highlights`, `GET /api/stats/summary`, `POST /api/detections/<id>/reclassify`.

### Web UI

**Detection detail page**: "Sound Detected" row in the details table, hidden when `sound_class` is null.

```
🔊 bird — Eurasian Blackbird (91.0%)   ← bird with BirdNET species
🔊 dog (82.0%)                          ← non-bird from PANNs
(row hidden)                            ← no animal sound detected
```

**Gallery page**: `🔊` badge overlaid on thumbnail cards when `sound_class` is set, showing species if available or the generic class.

## Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| Animal seen, no animal sound | `sound_class = NULL` — visual detection shown alone, no sound badge |
| Animal seen, animal sound matches | Both stored — strong corroboration |
| Animal seen, sound contradicts vision | Both stored independently — user sees both, no automatic override |
| No animal seen, animal sound present | Sound stored in DB; WAV retained; visual class remains "unknown" |
| No audio file (mic disabled/failed) | Audio classification skipped entirely |
| PANNs unavailable or disabled | Skip Pass 1; BirdNET runs unconditionally as standalone detector |
| BirdNET unavailable or disabled | Skip Pass 2; PANNs result used without species detail |
| Both unavailable | Feature silently disabled, logged as warning at startup |
| PANNs detects only generic "Animal" label | Used as fallback `sound_class = "animal"` when no specific category found |
| Crash between capture and mux/discard | Orphaned WAV cleaned up by `cleanup_detection()` on next cleanup run |

## Dependencies

| Package | Purpose | Size | Pi compatible |
|---------|---------|------|---------------|
| `panns_inference` | General audio classification | ~80 MB model (CNN14) | Yes (PyTorch) |
| `birdnet` | Bird species ID | ~20 MB model (TFLite) | Yes (ARM64 TFLite) |

Both depend on libraries already in the project (`torch` for PANNs, `numpy`/`soundfile` for BirdNET).

## What Stays Unchanged

- **Audio capture pipeline** — `AudioRecorder` and `audio_mux.py` are untouched
- **Visual classification** — runs independently as before
- **Object detection** — runs independently as before
- **Timelapse generation** — no audio analysis for timelapses
- **Video playback** — audio is already muxed into MP4

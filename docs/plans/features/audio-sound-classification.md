# Audio Sound Classification

## Overview

Classify animal sounds from audio captured alongside video. When a video is captured and an animal is visually recognised, the recorded audio is analysed to identify whether it contains an animal sound and, if so, what species or category it belongs to. Results are stored in the database and displayed on the detection and gallery pages.

## Prerequisites

- Audio capture feature is enabled and working (see `audio-capture.md`)
- `.wav` files are already recorded alongside each video via `AudioRecorder`

## Two-Pass Classification Strategy

Audio analysis uses two complementary models in sequence:

```
Audio file (.wav)
  │
  ├─► Pass 1: PANNs (broad) ──► "Is there an animal sound?" + category
  │                               (dog_bark, cat_meow, frog_croak, insect, bird, ...)
  │
  └─► Pass 2: BirdNET (specific) ─► Species-level bird ID
                                     (only runs when Pass 1 or visual classifier indicates "bird")
```

### Pass 1 — PANNs (general animal sound detection)

- **Model**: CNN14 from [panns_inference](https://github.com/qiuqiangkong/panns_inference) (PyTorch-based, already a project dependency)
- **Purpose**: Broad animal sound detection across all categories
- **Output**: AudioSet class label + confidence (e.g. "Dog bark — 0.82")
- **AudioSet animal classes to filter for**: dog, cat, frog, insect, bird, livestock, rodent, snake hiss, howl, growl, roar, etc.
- **Behaviour**: If no animal sound is found above the confidence threshold → `sound_class = NULL`, analysis stops. The visual detection stands on its own.

### Pass 2 — BirdNET (bird species identification)

- **Model**: BirdNET v2.4 via the [`birdnet`](https://github.com/birdnet-team/birdnet) Python package (TFLite backend)
- **Purpose**: Species-level identification for bird sounds
- **Trigger**: Only runs when Pass 1 detected a bird-category sound OR the visual classifier detected `animal_class = "bird"`
- **Output**: Species name + confidence (e.g. "Turdus merula_Eurasian Blackbird — 0.91")
- **Location filtering**: BirdNET supports lat/lng/week filters to narrow species predictions — reuse the `daylight.lat` / `daylight.lng` config values

## Implementation Steps

### Step 1 — Add dependencies

Add to `requirements.txt` (as optional, like torch):
```
panns-inference    # General audio classification (PyTorch)
birdnet            # Bird species identification (TFLite)
```

Both should degrade gracefully with `ImportError` handling.

### Step 2 — Create `src/analysis/audio_classifier.py`

New module following the same patterns as `classifier.py`:

```python
@dataclass
class AudioClassificationResult:
    sound_class: Optional[str]        # General category: "bird", "dog", "frog", "insect", None
    sound_species: Optional[str]      # Species-level ID (birds only): "Turdus merula_Eurasian Blackbird"
    sound_confidence: Optional[float] # Confidence of the best match
    all_predictions: list[tuple[str, float]]  # Top predictions for debugging

class AudioClassifier:
    def __init__(self, min_confidence: float = 0.5, lat: float = None, lng: float = None):
        ...

    def classify_audio(self, wav_path: Path, visual_animal_class: Optional[str] = None) -> AudioClassificationResult:
        """
        Two-pass audio classification.

        Args:
            wav_path: Path to the .wav file
            visual_animal_class: The animal class from visual classification (used to
                                 decide whether to run BirdNET even if PANNs is unsure)

        Returns:
            AudioClassificationResult with sound_class, sound_species, sound_confidence
        """
        # Pass 1: PANNs — broad animal sound detection
        panns_result = self._classify_panns(wav_path)

        if panns_result is None:
            return AudioClassificationResult(sound_class=None, ...)

        # Pass 2: BirdNET — only if bird-related
        if panns_result.is_bird or visual_animal_class == "bird":
            birdnet_result = self._classify_birdnet(wav_path)
            if birdnet_result:
                return AudioClassificationResult(
                    sound_class="bird",
                    sound_species=birdnet_result.species,
                    sound_confidence=birdnet_result.confidence,
                )

        # Non-bird animal sound from PANNs
        return AudioClassificationResult(
            sound_class=panns_result.category,
            sound_species=None,
            sound_confidence=panns_result.confidence,
        )
```

Key design points:
- Lazy model loading (models loaded on first call, not at init)
- Graceful degradation: if `panns_inference` not installed → skip Pass 1; if `birdnet` not installed → skip Pass 2
- Both passes can work independently (only PANNs, only BirdNET, or both)

### Step 3 — Database schema changes

Add three nullable columns to the `detections` table using the existing migration pattern (same as `bird_species` and `highlighted`):

```sql
ALTER TABLE detections ADD COLUMN sound_class TEXT;
ALTER TABLE detections ADD COLUMN sound_species TEXT;
ALTER TABLE detections ADD COLUMN sound_confidence REAL;
```

Update in `database.py`:
- Add fields to `Detection` dataclass: `sound_class`, `sound_species`, `sound_confidence`
- Add migration in `_init_db()` (check column existence, ALTER TABLE if missing)
- Update `_row_to_detection()` to read new columns
- Update `update_detection()` to accept new fields

### Step 4 — Integrate into `main.py` analysis pipeline

In `_process_capture()`, after video classification and object detection:

```python
# Audio classification (runs after visual analysis)
if self._audio_classifier and metadata.audio_path and metadata.audio_path.exists():
    try:
        # Get the visual animal class to help decide BirdNET pass
        current = self._database.get_detection(detection_id)
        visual_class = current.animal_class if current else None

        audio_result = self._audio_classifier.classify_audio(
            metadata.audio_path,
            visual_animal_class=visual_class,
        )

        if audio_result.sound_class:
            self._database.update_detection(
                detection_id,
                sound_class=audio_result.sound_class,
                sound_species=audio_result.sound_species,
                sound_confidence=audio_result.sound_confidence,
            )
            logger.info(
                f"Audio: {audio_result.sound_class} "
                f"({audio_result.sound_confidence:.1%})"
                + (f" — {audio_result.sound_species}" if audio_result.sound_species else "")
            )
        else:
            logger.info("No animal sound detected in audio")
    except Exception as e:
        logger.error(f"Audio classification failed: {e}")
```

Initialize the classifier in `_init_audio_classifier()` (new method), controlled by config:

```python
def _init_audio_classifier(self):
    audio_analysis_cfg = self._config_data.get("audio_analysis", {})
    if not audio_analysis_cfg.get("enabled", False):
        return

    try:
        from src.analysis.audio_classifier import AudioClassifier
        daylight_cfg = self._config_data.get("daylight", {})
        self._audio_classifier = AudioClassifier(
            min_confidence=audio_analysis_cfg.get("min_confidence", 0.5),
            lat=daylight_cfg.get("lat"),
            lng=daylight_cfg.get("lng"),
        )
        logger.info("Audio classifier initialized (PANNs + BirdNET)")
    except ImportError as e:
        logger.warning(f"Audio classification dependencies not available: {e}")
        self._audio_classifier = None
```

### Step 5 — Configuration

Add to `config.yaml.example`:

```yaml
# Audio sound classification
audio_analysis:
  enabled: true
  min_confidence: 0.5          # Minimum confidence to store a result
  panns_enabled: true           # General animal sound detection
  birdnet_enabled: true         # Bird species identification
```

BirdNET location filtering reuses the existing `daylight.lat` / `daylight.lng` values — no duplicate config needed.

### Step 6 — API changes

Expose new fields in all detection API responses in `api.py`. Add to every detection serialisation dict:

```python
"sound_class": d.sound_class,
"sound_species": d.sound_species,
"sound_confidence": d.sound_confidence,
```

Affected endpoints:
- `GET /api/detections` (list)
- `GET /api/detections/<id>` (detail)
- `GET /api/highlights` (list)
- `GET /api/summary` (recent detections)

Add optional query parameter to `GET /api/detections`:
- `has_sound=true` — filter to only detections with `sound_class IS NOT NULL`

### Step 7 — Web UI changes

#### Detection detail page
When `sound_class` is present, show a sound badge below the visual classification:

```
🔊 Bird call — Eurasian Blackbird (91%)     ← bird with species from BirdNET
🔊 Dog bark (82%)                            ← non-bird from PANNs
(no badge)                                   ← no animal sound detected
```

#### Gallery page
- Add a small 🔊 icon overlay on detection cards that have `sound_class` set
- Optional: add a "Has sound" filter toggle in the filter bar

## Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| Animal seen, no animal sound | `sound_class = NULL` — visual detection shown alone, no sound badge |
| Animal seen, animal sound matches | Both shown — strong corroboration |
| Animal seen, sound contradicts vision | Both stored independently — user sees both, no automatic override |
| No animal seen, animal sound present | Audio classification still runs but result is stored; visual class remains "unknown" |
| No audio file (mic disabled/failed) | Audio classification skipped entirely — no change to current behaviour |
| PANNs unavailable (not installed) | Skip Pass 1; BirdNET can still run if visual class is "bird" |
| BirdNET unavailable (not installed) | Skip Pass 2; PANNs result used for `sound_class` without species detail |
| Both unavailable | Feature silently disabled, logged as warning at startup |

## Dependencies

| Package | Purpose | Size | Pi compatible |
|---------|---------|------|---------------|
| `panns_inference` | General audio classification | ~80MB model (CNN14) | Yes (PyTorch, already used) |
| `birdnet` | Bird species ID | ~20MB model (TFLite) | Yes (ARM64 TFLite) |

Both depend on libraries already in the project (`torch` for PANNs, `numpy`/`soundfile` for BirdNET). No new system-level dependencies required.

## Data Directories

No new directories needed. Audio `.wav` files are already captured to `data/videos/` and muxed into the `.mp4`. The classification results live in the database only.

## What Stays Unchanged

- **Audio capture pipeline** — `AudioRecorder` and `audio_mux.py` are untouched
- **Visual classification** — runs independently as before
- **Object detection** — runs independently as before
- **Timelapse generation** — no audio analysis for timelapses
- **Video playback** — audio is already muxed into MP4

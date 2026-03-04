# Wake Word Improvement Plan

**Issue:** [#166](https://github.com/...) -- Improve wake word detection reliability
**Date:** 2026-03-04
**Status:** Research complete, ready for implementation

---

## 1. Current Implementation

### Model

| Property | Value |
|----------|-------|
| Model file | `devices/esp32-s3-box3-agent/firmware/models/hi_roo.tflite` |
| Header embed | `devices/esp32-s3-box3-agent/firmware/main/hi_roo_model.h` (5078 lines) |
| Model size | 60,896 bytes (~60KB) |
| Architecture | MixedNet (microWakeWord streaming model) |
| Quantization | INT8 (full integer quantization) |
| Training framework | [microWakeWord](https://github.com/kahrendt/microWakeWord) |
| Training data | ~1500 synthetic samples via Piper TTS, 904 speaker embeddings |
| Reported accuracy | prob=0.941 on initial test |
| Ops used | 13: Conv2D, DepthwiseConv2D, FullyConnected, Reshape, Logistic, Quantize, StridedSlice, Concatenation, SplitV, VarHandle, ReadVariable, AssignVariable, CallOnce |

### Detector Code

**File:** `devices/esp32-s3-box3-agent/firmware/main/elisa_wake_word.cc`

The detector uses streaming inference with these key parameters:

| Parameter | Value | Location (line) |
|-----------|-------|-----------------|
| `kSampleRate` | 16000 Hz | L40 |
| `kFeatureCount` | 40 mel channels | L41 |
| `kStrideSizeMs` | 10ms | L42 |
| `kStrideSamples` | 160 samples | L43 |
| `kModelInputFrames` | 3 frames per inference | L44 |
| `kProbabilityCutoff` | **0.94** | L45 |
| `kSlidingWindowSize` | **5 frames** | L46 |
| `kMinSlicesBeforeDetect` | 74 (~740ms) | L47 |
| `kTensorArenaSize` | 65536 bytes (64KB) | L48 |
| `kNumResourceVariables` | 6 streaming state buffers | L49 |

### Audio Feature Extraction (L79-L108)

Uses ESPMicroSpeechFeatures (`frontend.h`, `frontend_util.h`) matching the microWakeWord training pipeline:

- 40-channel mel filterbank (125 Hz - 7500 Hz)
- 30ms window, 10ms stride
- PCAN gain control (strength=0.95, offset=80.0, gain_bits=21)
- Noise reduction (smoothing_bits=10, even=0.025, odd=0.06, min_signal=0.05)
- Log scaling (scale_shift=6)
- Quantization: `int8 = ((uint16_feature * 256 + 333) / 666) - 128`

### Detection Logic (L193-L278)

1. Accumulate audio into 160-sample stride buffers
2. Extract mel spectrogram via `FrontendProcessSamples()`
3. Fill ring buffer of 3 frames (kModelInputFrames)
4. Run TFLite inference every frame once buffer is full
5. Read uint8 output, scale to `float probability = raw / 255.0`
6. Update sliding window of 5 probabilities
7. After 74 slices (~740ms), check if `mean(window) >= 0.94`
8. Return `true` on detection

### Integration with ESP-SR

The TFLite wake word detector exists alongside ESP-SR. In `elisa_main.c:248-253`, `init_wake_word()` calls `app_sr_start(false)` which initializes ESP-SR's WakeNet. The commit message for `3364b27` notes: "TFLite detector runs first, falls back to WakeNet ('Hi ESP') if init fails."

Currently the device.json Blockly block only exposes ESP-SR wake words (hey_elisa, hey_box, hi_alex, hey_computer). The custom "Hi Roo" model is a parallel addition not yet surfaced in the UI.

---

## 2. Problems with Current Model

### 2.1 Training Data Limitations

- **Synthetic only:** All 1500 training samples are Piper TTS-generated. No real human voice recordings.
- **Limited speaker diversity:** While 904 speaker embeddings were used for Piper synthesis, TTS voices have a narrower acoustic range than real humans (no children's voices, no accented speech, no whispering, no shouting).
- **No environmental diversity:** Trained in clean audio conditions. No background noise, no room reverb, no distance variation.
- **No negative mining:** No targeted training against phonetically similar phrases that should NOT trigger the wake word.

### 2.2 Detection Parameters May Be Suboptimal

- **Probability cutoff 0.94** was derived from a training ROC curve with synthetic data. May be too aggressive or too lenient for real-world use.
- **Sliding window of 5 frames** (50ms at 10ms stride) is relatively small. Increasing it would smooth out noise spikes but add latency.
- **Min slices before detect = 74** (~740ms) prevents very early false triggers but doesn't help with ongoing false positive/negative rates.

### 2.3 Model Architecture Constraints

The 60KB model budget is reasonable for ESP32-S3 but limits the MixedNet depth. The current model may not have enough capacity to discriminate between "Hi Roo" and similar-sounding phrases.

---

## 3. Recommended Improvements

### 3.1 Training Data Diversity

**Priority: High | Effort: Medium (data collection takes time)**

#### Positive Samples (target: 5000+ total, up from 1500)

| Source | Count | Notes |
|--------|-------|-------|
| Keep existing Piper TTS | 1500 | Retain as baseline |
| Real voice recordings (adults) | 1000+ | Multiple speakers, distances (0.5m, 1m, 2m, 3m), speaking styles (casual, loud, quiet, fast, slow) |
| Real voice recordings (children ages 8-14) | 500+ | Primary user demographic |
| Speed/pitch-augmented real recordings | 1000+ | +/- 10% speed, +/- 2 semitone pitch |

#### Recording Conditions

Capture real samples across these conditions:

| Condition | Variants |
|-----------|----------|
| Distance | 0.5m, 1m, 2m, 3m from BOX-3 microphone |
| Background noise | Quiet room, TV playing, music, multiple people talking, kitchen sounds |
| Room acoustics | Small room (bedroom), large room (living room), kitchen (hard surfaces) |
| Speaker age | Children (8-14), teens, adults (male, female) |
| Speaking style | Normal, whispered, shouted, fast, slow, sleepy |

### 3.2 Noise Augmentation Strategy

**Priority: High | Effort: Low (automated pipeline)**

Apply data augmentation to all clean recordings before training:

| Augmentation | Parameters |
|--------------|------------|
| Additive noise (white) | SNR 5dB, 10dB, 15dB, 20dB |
| Additive noise (babble) | SNR 5dB, 10dB, 15dB |
| Additive noise (household) | TV, music, kitchen; SNR 5-15dB |
| Room impulse response (RIR) | Small room, medium room, large room convolutions |
| Volume variation | -6dB to +6dB random gain |
| Speed perturbation | 0.9x to 1.1x playback speed |
| Pitch shift | -2 to +2 semitones |

Use a tool like [audiomentations](https://github.com/iver56/audiomentations) or [SoX](https://sox.sourceforge.net/) for batch augmentation.

### 3.3 Negative Mining

**Priority: High | Effort: Medium**

Train the model to reject phonetically similar phrases. Collect/synthesize negative samples for:

| Category | Examples |
|----------|----------|
| Phonetically similar | "hero", "hi room", "Peru", "hi boo", "hieroglyphic", "high roof", "hi Ruth" |
| Partial matches | "hi", "roo", "kangaroo", "wahoo", "taboo", "voodoo" |
| Common household phrases | "hey Google", "hey Siri", "Alexa", "hey Elisa", "hello", "hi there" |
| Background speech | General conversation snippets, TV/radio audio |
| Non-speech | Coughing, laughing, clapping, door closing, phone ringing |

Target: 5000+ negative samples with the same augmentation pipeline applied.

### 3.4 Threshold and Window Tuning

**Priority: High | Effort: Low (code changes only)**

These parameters in `elisa_wake_word.cc` are adjustable without retraining the model:

#### Probability Cutoff (`kProbabilityCutoff`, line 45)

| Value | Effect |
|-------|--------|
| 0.90 | More sensitive (fewer missed detections, more false positives) |
| **0.94** | **Current value** |
| 0.96 | More selective (fewer false positives, more missed detections) |
| 0.98 | Very selective (almost no false positives, may miss quiet/distant speech) |

**Recommendation:** Test values between 0.90 and 0.97 in real-world conditions. The optimal value depends heavily on the retrained model's confidence distribution. Start with 0.92 for a more forgiving experience during development.

#### Sliding Window Size (`kSlidingWindowSize`, line 46)

| Value | Latency Added | Effect |
|-------|--------------|--------|
| 3 | ~30ms | Faster response, more noise sensitivity |
| **5** | **~50ms** | **Current value** |
| 7 | ~70ms | Smoother, fewer transient false triggers |
| 10 | ~100ms | Very smooth, noticeable detection latency |

**Recommendation:** Try 7 for a better balance. The added 20ms latency is imperceptible, but averaging over more frames significantly reduces transient false positives.

#### Minimum Slices Before Detect (`kMinSlicesBeforeDetect`, line 47)

Current value 74 (~740ms) sets how long after a reset before detection is possible. This prevents immediate re-triggering after the previous detection.

**Recommendation:** Keep at 74. This is already reasonable for preventing re-triggers during the TTS playback tail.

#### Consecutive High-Probability Requirement (not currently implemented)

**New feature:** Instead of just averaging the sliding window, require N consecutive frames above a lower threshold before accepting. This would reject brief noise spikes more effectively.

```c
// Proposed: require 3+ consecutive frames above 0.85 within the window
static constexpr float kConsecutiveThreshold = 0.85f;
static constexpr int kMinConsecutiveFrames = 3;
```

This is a stronger filter than mean-only and could significantly reduce false positives from brief audio events.

### 3.5 Model Architecture Options (within 60KB budget)

**Priority: Medium | Effort: High (requires retraining)**

| Architecture | Approx Size | Notes |
|--------------|-------------|-------|
| **MixedNet (current)** | ~60KB | microWakeWord default. Good balance of accuracy and size. |
| MixedNet with increased depth | ~60KB | Use narrower channels but more layers. May improve discrimination. |
| Depthwise separable CNN | ~40-50KB | Lighter, more room for larger sliding window or dual-model voting. |
| Multi-head output | ~65KB | Single model with "Hi Roo" head + "not Hi Roo" head. Better calibrated probabilities. |

**Recommendation:** Stick with MixedNet for now. The microWakeWord framework is well-optimized for it, and the training data improvements will yield much larger gains than architecture changes at this model size.

---

## 4. Backend/UI Integration Notes

### Current State

- `device.json` exposes only ESP-SR wake words in the Blockly dropdown: hey_elisa, hey_box, hi_alex, hey_computer
- The "Hi Roo" TFLite model runs alongside ESP-SR but is not user-selectable
- `flashStrategy.ts:302` passes `wake_word` from the Blockly field into `runtime_config.json`
- `redeployClassifier.ts` correctly identifies wake_word as a firmware-level change (requires reflash)

### Future Work (out of scope for this task)

- Add "Hi Roo" (and potentially other custom wake words) to the Blockly dropdown once the model is production-ready
- Update `init_wake_word()` in `elisa_main.c` to select between ESP-SR and TFLite based on the configured wake word
- Consider a hybrid approach: ESP-SR for built-in wake words, TFLite for custom ones

---

## 5. Concrete Next Steps

### Phase 1: Quick Wins (code-only, no retraining)

| Step | Effort | Expected Impact |
|------|--------|----------------|
| 1. Lower `kProbabilityCutoff` from 0.94 to 0.92 | 1 line change | Fewer missed detections at the cost of slightly more false positives |
| 2. Increase `kSlidingWindowSize` from 5 to 7 | 1 line change | Smoother detection, fewer transient false triggers |
| 3. Add consecutive-frame requirement (3+ frames above 0.85) | ~20 lines | Stronger false positive rejection |
| 4. Field test with 3-5 people at various distances | 2-3 hours | Establish baseline metrics (detection rate, false positive rate) |

### Phase 2: Training Data Collection

| Step | Effort | Expected Impact |
|------|--------|----------------|
| 5. Record 500+ real "Hi Roo" samples (adults + kids) | 1-2 days | Real voice data dramatically improves robustness |
| 6. Record 500+ negative samples (similar phrases + background) | 1-2 days | Reduces false positives on confusable phrases |
| 7. Build noise augmentation pipeline (audiomentations) | 2-4 hours | Multiplies training data 5-10x with realistic conditions |
| 8. Apply augmentation to all samples | 1 hour (automated) | 5000+ augmented training samples |

### Phase 3: Retrain and Validate

| Step | Effort | Expected Impact |
|------|--------|----------------|
| 9. Retrain MixedNet with expanded dataset | 2-4 hours (GPU) | Significantly improved model |
| 10. Generate new ROC curve, select optimal threshold | 1 hour | Data-driven threshold instead of single-test guess |
| 11. Convert to TFLite, verify size <= 60KB | 30 min | Ensure model fits ESP32-S3 constraints |
| 12. Replace `hi_roo.tflite` and `hi_roo_model.h` | 30 min | Deploy improved model |
| 13. Field test retrained model (same protocol as step 4) | 2-3 hours | Measure improvement vs baseline |

### Phase 4: Production Readiness

| Step | Effort | Expected Impact |
|------|--------|----------------|
| 14. A/B test threshold values on retrained model | 1-2 days | Find optimal operating point |
| 15. Add "Hi Roo" to Blockly wake word dropdown | 1-2 hours | Make custom wake word user-selectable |
| 16. Update `init_wake_word()` for TFLite/ESP-SR routing | 2-4 hours | Clean integration of both wake word systems |
| 17. Document custom wake word training process | 2-4 hours | Enable future custom wake words |

---

## 6. Key Files Reference

| File | Purpose |
|------|---------|
| `devices/esp32-s3-box3-agent/firmware/main/elisa_wake_word.cc` | TFLite detector implementation |
| `devices/esp32-s3-box3-agent/firmware/main/elisa_wake_word.h` | Public API (init, detect, reset, cleanup) |
| `devices/esp32-s3-box3-agent/firmware/main/hi_roo_model.h` | Embedded model as C byte array (5078 lines) |
| `devices/esp32-s3-box3-agent/firmware/models/hi_roo.tflite` | Source TFLite model file (60,896 bytes) |
| `devices/esp32-s3-box3-agent/firmware/main/elisa_main.c` | Boot sequence, `init_wake_word()` at L248 |
| `devices/esp32-s3-box3-agent/device.json` | Plugin manifest with wake word dropdown options |
| `backend/src/services/flashStrategy.ts` | Wake word config flow (L302) |
| `backend/src/services/redeployClassifier.ts` | Classifies wake_word as firmware-level change |

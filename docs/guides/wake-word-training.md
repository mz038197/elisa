# Wake Word Training Tutorial

Train a custom "Hi Roo" wake word model for the ESP32-S3-BOX-3. This guide takes you from raw recordings to a deployed model.

## Prerequisites

- ESP32-S3-BOX-3 connected via USB-C
- Python 3.8+ with pip
- Google account (for Colab training)
- ~2 hours for recording, ~30 min for augmentation, ~2-4 hours for training

## Directory Layout

```
devices/esp32-s3-box3-agent/wake-word-training/
  record.py          # Record samples from BOX-3 mic
  augment.py         # Generate augmented training data
  train.ipynb        # Google Colab notebook for model training
  deploy.sh          # Install trained model into firmware
  requirements.txt   # Python dependencies
  data/
    positive/        # Raw "Hi Roo" recordings
    negative/        # Raw non-wake-word recordings
    augmented_positive/   # Augmented positive samples
    augmented_negative/   # Augmented negative samples
```

## Step 1: Install Dependencies

```bash
cd devices/esp32-s3-box3-agent/wake-word-training
pip install -r requirements.txt
```

## Step 2: Record Positive Samples (the wake word)

Connect your BOX-3 via USB-C. The recording script puts the device in recording mode and captures 2-second clips through its actual microphones.

```bash
# Record yourself
python record.py --speaker dad --mode positive

# Record a kid
python record.py --speaker kid1 --mode positive
```

The script guides you through prompts:
```
  [001] Say 'Hi Roo' in your normal voice
  Press ENTER to start recording (or 'q' to stop):
  Saved: hiroo_dad_001.wav (64000 bytes)
  Another? [Y/n/q]:

  [002] Say 'Hi Roo' a bit louder
  Press ENTER to start recording (or 'q' to stop):
  ...
```

**Tips for good recordings:**
- Record at least 50 samples per speaker (more is better)
- Vary your distance: hold the BOX-3 at arm's length, set it on a table 1m away, across the room at 3m
- Vary your volume: normal, quiet (like a whisper), loud
- Vary your speed: normal, fast, slow, sleepy
- Record in different rooms (bedroom, kitchen, living room)

**Target: 200+ positive samples across all speakers.**

### No BOX-3 handy? Use your laptop mic

```bash
python record.py --speaker dad --mode positive --laptop
```

This uses your laptop/desktop microphone instead. Recordings won't perfectly match the BOX-3's dual MEMS mics, but augmentation compensates for most of the difference.

## Step 3: Record Negative Samples

These teach the model what NOT to trigger on. Equally important as positives.

```bash
python record.py --speaker dad --mode negative
```

The script prompts you through similar-sounding phrases and background audio:
```
  [001] Say 'hero'
  [002] Say 'hi room'
  [003] Say 'Peru'
  ...
  [012] Just stay quiet (silence)
  [013] Talk normally about anything (background speech)
  [014] Clap your hands a few times
```

**Tips:**
- Record every confusable phrase multiple times at different volumes/distances
- Include ambient noise: TV on, music playing, kitchen sounds
- Record silence in each room you use
- Have kids say the negative phrases too

**Target: 200+ negative samples across all speakers.**

## Step 4: Check Your Data

```bash
ls data/positive/*.wav | wc -l   # Should be 200+
ls data/negative/*.wav | wc -l   # Should be 200+
```

If you have fewer than 100 of either, record more. The model quality directly scales with data quantity.

## Step 5: Augment

The augmentation script takes each recording and generates ~10 variants with realistic distortions (noise, reverb, speed changes, volume variation).

```bash
python augment.py
```

Output:
```
Input: 250 positive, 250 negative raw samples
Multiplier: 10x (will produce ~2750 positive, ~2750 negative)

Augmenting positive samples...
  -> 2750 augmented positive samples
Augmenting negative samples...
  -> 2750 augmented negative samples
```

For more or fewer variants:
```bash
python augment.py --multiplier 5    # 5x (faster, smaller dataset)
python augment.py --multiplier 20   # 20x (slower, larger dataset)
```

## Step 6: Package for Upload

Create a zip file for Colab:

```bash
cd wake-word-training
zip -r training_data.zip data/augmented_positive data/augmented_negative
```

This file will be ~500MB-1GB depending on sample count.

## Step 7: Train in Google Colab

1. Open `train.ipynb` in Google Colab:
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File > Upload notebook > select `train.ipynb`

2. **Set runtime to GPU:**
   - Runtime > Change runtime type > T4 GPU

3. **Run all cells** (Runtime > Run all), which will:
   - Install microWakeWord and TensorFlow
   - Prompt you to upload `training_data.zip`
   - Extract and load the augmented audio
   - Extract mel spectrograms (matching firmware config)
   - Train a MixedNet model (~100 epochs, early stopping)
   - Show ROC curve and recommend a detection threshold
   - Export INT8-quantized TFLite model
   - Download `hi_roo.tflite`

4. **Review the ROC curve** (cell 5):
   - Blue line should hug the top-left corner (high AUC)
   - The threshold chart shows detection rate vs false positive rate
   - The notebook recommends a threshold where FPR < 1% and detection > 90%
   - Note the recommended threshold — you'll use it in the firmware

## Step 8: Deploy to Firmware

Move the downloaded `hi_roo.tflite` to the training directory, then:

```bash
./deploy.sh hi_roo.tflite
```

This:
1. Copies the model to `firmware/models/hi_roo.tflite`
2. Generates `firmware/main/hi_roo_model.h` (C byte array)

### Update the threshold (if needed)

If the Colab notebook recommended a threshold different from 0.92, edit `firmware/main/elisa_wake_word.cc`:

```c
static constexpr float kProbabilityCutoff = 0.XX;  // from Colab ROC curve
```

### Rebuild and flash

```bash
cd devices/esp32-s3-box3-agent
./build-firmware.sh
```

Then flash via Elisa (deploy pipeline) or manually:

```bash
esptool.py --chip esp32s3 --port /dev/cu.usbmodem* --baud 460800 \
  write_flash 0x10000 firmware/box3-agent.bin
```

## Step 9: Test

Say "Hi Roo" at various distances and volumes. Watch the serial monitor:

```bash
idf.py -p /dev/cu.usbmodem* monitor
```

You should see:
```
I (12345) elisa_wake: Wake word detected! prob=0.953 consec=5
```

### If detection is too sensitive (false positives)
- Raise `kProbabilityCutoff` by 0.02 (e.g., 0.92 → 0.94)
- Raise `kMinConsecutiveFrames` from 3 to 4

### If detection misses too often
- Lower `kProbabilityCutoff` by 0.02 (e.g., 0.92 → 0.90)
- Lower `kMinConsecutiveFrames` from 3 to 2
- Record more samples and retrain

## Quick Reference

| Command | What it does |
|---------|-------------|
| `python record.py --speaker dad` | Record positive + negative samples |
| `python record.py --mode positive --laptop` | Record positives using laptop mic |
| `python augment.py` | Generate 10x augmented training data |
| `zip -r training_data.zip data/augmented_*` | Package for Colab upload |
| `./deploy.sh hi_roo.tflite` | Install trained model into firmware |
| `./deploy.sh hi_roo.tflite --rebuild` | Install and rebuild firmware |

## Iterating

The model improves with more data. After testing, you can:

1. Record more samples in conditions where detection failed
2. Re-run `augment.py` (it processes all samples in `data/positive/` and `data/negative/`)
3. Re-zip and re-upload to Colab
4. Retrain and redeploy

Each iteration takes ~1 hour once you have the workflow down.

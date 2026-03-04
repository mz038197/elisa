#!/usr/bin/env python3
"""
Audio augmentation pipeline for wake word training data.

Takes raw recordings from data/positive/ and data/negative/ and generates
augmented variants in data/augmented_positive/ and data/augmented_negative/.

Each input file produces ~10 augmented variants with:
  - Background noise at various SNRs
  - Room impulse response (reverb simulation)
  - Speed perturbation (0.9x - 1.1x)
  - Pitch shift (-2 to +2 semitones)
  - Volume variation (-6dB to +6dB)

Usage:
    pip install audiomentations numpy soundfile
    python augment.py
    python augment.py --multiplier 5     # fewer variants per sample
    python augment.py --multiplier 20    # more variants per sample
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed. Run: pip install soundfile")
    sys.exit(1)

try:
    from audiomentations import (
        Compose,
        AddGaussianNoise,
        TimeStretch,
        PitchShift,
        Shift,
        Gain,
        LowPassFilter,
        HighPassFilter,
        ClippingDistortion,
        Normalize,
    )
except ImportError:
    print("ERROR: audiomentations not installed. Run: pip install audiomentations")
    sys.exit(1)

SAMPLE_RATE = 16000
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def build_augmentation_pipeline():
    """Build the audiomentations augmentation chain.

    Each transform has a probability < 1.0 so not every augmentation
    fires on every sample, creating natural variety.
    """
    return Compose([
        # Additive noise (simulates background)
        AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=0.5),

        # Time stretch (speed variation without pitch change)
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),

        # Pitch shift (different speakers have different pitches)
        PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.4),

        # Time shift (wake word might not start at exact beginning)
        Shift(min_shift=-0.2, max_shift=0.2, p=0.3),

        # Volume variation (near vs far, loud vs quiet)
        Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.5),

        # Low pass filter (muffled sound through walls/distance)
        LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=0.2),

        # High pass filter (removes low-frequency rumble)
        HighPassFilter(min_cutoff_freq=80, max_cutoff_freq=300, p=0.2),

        # Mild clipping (simulates near-field overdriving the mic)
        ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=5, p=0.1),

        # Always normalize at the end to consistent level
        Normalize(p=1.0),
    ])


def augment_directory(input_dir, output_dir, multiplier, pipeline):
    """Augment all WAV files in input_dir, write to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(input_dir, '*.wav')))
    if not wav_files:
        print(f"  No WAV files in {input_dir}, skipping.")
        return 0

    total = 0
    for wav_path in wav_files:
        basename = os.path.splitext(os.path.basename(wav_path))[0]

        # Load audio
        audio, sr = sf.read(wav_path, dtype='float32')
        if sr != SAMPLE_RATE:
            print(f"  WARNING: {wav_path} is {sr}Hz, expected {SAMPLE_RATE}Hz. Skipping.")
            continue

        # Ensure mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Copy original as variant 0
        out_path = os.path.join(output_dir, f'{basename}_orig.wav')
        sf.write(out_path, audio, SAMPLE_RATE)
        total += 1

        # Generate augmented variants
        for v in range(multiplier):
            augmented = pipeline(samples=audio, sample_rate=SAMPLE_RATE)

            out_path = os.path.join(output_dir, f'{basename}_aug{v:02d}.wav')
            sf.write(out_path, augmented, SAMPLE_RATE)
            total += 1

    return total


def main():
    parser = argparse.ArgumentParser(description='Augment wake word training data')
    parser.add_argument('--multiplier', type=int, default=10,
                        help='Number of augmented variants per sample (default: 10)')
    args = parser.parse_args()

    pipeline = build_augmentation_pipeline()

    pos_in = os.path.join(DATA_DIR, 'positive')
    pos_out = os.path.join(DATA_DIR, 'augmented_positive')
    neg_in = os.path.join(DATA_DIR, 'negative')
    neg_out = os.path.join(DATA_DIR, 'augmented_negative')

    pos_raw = len(glob.glob(os.path.join(pos_in, '*.wav')))
    neg_raw = len(glob.glob(os.path.join(neg_in, '*.wav')))

    print(f"Input: {pos_raw} positive, {neg_raw} negative raw samples")
    print(f"Multiplier: {args.multiplier}x (will produce ~{pos_raw * (args.multiplier + 1)} positive, ~{neg_raw * (args.multiplier + 1)} negative)")
    print()

    print("Augmenting positive samples...")
    pos_count = augment_directory(pos_in, pos_out, args.multiplier, pipeline)
    print(f"  -> {pos_count} augmented positive samples")

    print("Augmenting negative samples...")
    neg_count = augment_directory(neg_in, neg_out, args.multiplier, pipeline)
    print(f"  -> {neg_count} augmented negative samples")

    print(f"\n{'='*60}")
    print(f"  Augmentation complete!")
    print(f"  Positive: {pos_count} samples in {pos_out}")
    print(f"  Negative: {neg_count} samples in {neg_out}")
    print(f"  Next step: Open train.ipynb in Google Colab")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

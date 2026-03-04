/**
 * @file elisa_wake_word.h
 * @brief Custom wake word detection using TFLite Micro.
 *
 * Replaces ESP-SR WakeNet with a microWakeWord-trained TFLite model.
 * The model runs streaming inference on 40-channel mel spectrograms
 * extracted from 16kHz mono PCM audio.
 */

#ifndef ELISA_WAKE_WORD_H
#define ELISA_WAKE_WORD_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the TFLite wake word detector.
 * Loads the embedded model and allocates the tensor arena in PSRAM.
 *
 * @return 0 on success, -1 on error
 */
int elisa_wake_word_init(void);

/**
 * Process a chunk of 16kHz mono PCM audio and check for wake word.
 *
 * Call this repeatedly with audio frames. Internally accumulates samples,
 * generates spectrograms every 10ms stride, runs inference, and applies
 * sliding window averaging.
 *
 * @param audio   16-bit signed PCM samples at 16kHz
 * @param samples Number of samples (not bytes)
 * @return true if wake word detected in this chunk
 */
bool elisa_wake_word_detect(const int16_t *audio, size_t samples);

/**
 * Reset the detector state (clear sliding window, feature buffers).
 * Call after detection to prepare for next wake word.
 */
void elisa_wake_word_reset(void);

/**
 * Clean up and free resources.
 */
void elisa_wake_word_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* ELISA_WAKE_WORD_H */

#ifndef __VNR_PRIV_H__
#define __VNR_PRIV_H__

#include "vnr_features_state.h"

/** Exponent of VNR input data. 
 * NOT USER MODIFIABLE.
 */
#define VNR_INPUT_EXP (-31)

/** Extra 2 samples you need to allocate in time domain so that the full spectrum (DC to nyquist) can be stored
 * after the in-place FFT. NOT USER MODIFIABLE.
 */  
#define VNR_FFT_PADDING (2)

/** Exponent of output of the log2 function used in the VNR
 */
#define VNR_LOG2_OUTPUT_EXP (-24)

/**
 * @brief Gain applied to the first mel filtered bin when highpass filtering of MEL output is enabled
 */
#define VNR_MEL_HP_GAIN (f32_to_float_s32((float)0.01))

// Matching with python names
/**
 * @brief Convert a spectrum to Mel frequency spectrum
 * This function passes the input DFT spectrum through a Mel filterbank consisting of VNR_MEL_FILTERS filters with overlapping bands, and
 * also does a log2 computation on the Mel filter output.
 *
 * @param[out] new_slice array containing VNR_MEL_FILTERS Mel+log2 output values. The output fixed point format is fixed to Q8.24.
 * @param[in] X Pointer to BFP structure containing the DFT output
 * @param[in] Flag to enable highpass filtering of MEL filtering output. When enabled, the first MEL output bin is scaled by VNR_MEL_HP_GAIN. 
 *
 * This function name matches with the corresponding function in py_voice python model.
*/
void vnr_priv_make_slice(uq8_24 *new_slice, const bfp_complex_s32_t *X, int32_t hp);

/**
 * @brief roll a buffer and add a new slice to the end
 * This function adds the new slice created in vnr_priv_make_slice() to the buffer holding the most recent VNR_PATCH_WIDTH slices that make up the
 * VNR_PATCH_WIDTH * VNR_MEL_FILTERS set of features that the inference engine runs on. 
 *
 * @param[inout] feature_buffers pointer to the feature patch buffer that is updated with the newest slice.
 * @param[in] new_slice New slice corresponding to the latest frame that is computed in vnr_priv_make_slice()
 *
 * This function name matches with the corresponding function in py_voice python model.
 */
void vnr_priv_add_new_slice(int32_t (*feature_buffers)[VNR_MEL_FILTERS], const int32_t *new_slice);

/**
 * @brief Normalise a patch by subtracting the max.
 * This function normalises the patch by subtracting the maximum value in the patch from every value in the patch.
 * The normalisation is not done in-place in the feature patch buffer as renormalisation is required for 
 * each slice added, so only the unnormalised features are buffered.
 *
 * @param[out] normalised_patch pointer to bfp_s32_t structure holding the normalised patch. The caller of this function needs to allocate this structure but
 *             doesn't need to initialise it.
 * @param[out] normalised_patch_data Pointer to int32 array of size VNR_PATCH_WIDTH*VNR_MEL_FILTERS for returning the normalised patch in.
 * @param[in] feature_state Pointer to the vnr_feature_state_t feature structure to read the patch buffer from. 
 *
 * This function name matches with the corresponding function in py_voice python model.
 */
void vnr_priv_normalise_patch(bfp_s32_t *normalised_patch, int32_t *normalised_patch_data, const vnr_feature_state_t *feature_state);

/**
 * @brief Calculate Discrete Fourier Transform (DFT) spectrum of an input time domain vector.
 *
 * This function calculates the spectrum of a real 32bit time domain vector. It calculates an N point real DFT where N is the length of the input vector to output a complex N/2+1 length complex 32bit vector.
 * The N/2+1 complex output values represent spectrum samples from DC up to the Nyquist frequency.
 *
 * The DFT calculation is done in place. After this function call, the output bfp_complex_s32_t BFP structure's `data` fields point to the input
 * time domain vector's memory.
 *
 * @param[out] X    DFT output BFP structure
 * @param[in] x_data input time domain vectory
 *
 * To allow for inplace transform from N real 32bit values to N/2+1 complex 32bit values, the input vector should have 2 extra real 32bit samples worth of memory.
 *
 */
void vnr_priv_forward_fft(bfp_complex_s32_t *X, int32_t *x_data);

/**
 * @brief Convert a spectrum to Mel frequency spectrum. 
 * The Mel filters are stored in the mel_filter_512_24_compact.h file that is autogenerated from the gen_mel_filters.py script.
 * The filters are stored in a compact form, meaning,
 * - Only the even filters are stored.
 * - Only the non-zero (and one zero coeff to indicate filter start) coefficients of every filter are stored in 1 array.
 * - The start bin of all even and odd filters are stored and using this, the odd filters are generated by 1 - even_filters at runtime in the vnr_priv_mel_compute() function
 *
 * @param[out] filter_output MEL filtering output spectrum output as VNR_MEL_FILTERS float_s32_t values.
 * @param[in] X DFT spectrum
 */
void vnr_priv_mel_compute(float_s32_t *filter_output, const bfp_complex_s32_t *X);

/**
 * @brief log2 operation on float_s32 values
 * This function takes in an array of float_s32_t values and performs element wise log2 operation on them.
 * The log2 output format is fixed to Q8.24.
 *
 * @param[out] output_q24 pointer to array that'll hold the log2 values.
 * @param[input] input float_s32_t values
 * @param[input] length length of the input array
 */
void vnr_priv_log2(uq8_24 *output_q24, const float_s32_t *input, unsigned length);

/**
 * @brief log2 of a single value
 * This function calculates output = log2(input) for a single float_s32_t input value. log2 output format is fixed to Q8.24
 */
uq8_24 vnr_priv_float_s32_to_fixed_q24_log2(float_s32_t x);

#endif

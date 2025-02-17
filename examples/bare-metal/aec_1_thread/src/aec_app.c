// Copyright 2022 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#include "aec_api.h"

#include "aec_config.h"
#include "aec_memory_pool.h"
#include "fileio.h"
#include "wav_utils.h"


extern void aec_process_frame_2threads(
        aec_state_t *main_state,
        aec_state_t *shadow_state,
        int32_t (*output_main)[AEC_FRAME_ADVANCE],
        int32_t (*output_shadow)[AEC_FRAME_ADVANCE],
        const int32_t (*y_data)[AEC_FRAME_ADVANCE],
        const int32_t (*x_data)[AEC_FRAME_ADVANCE]);

void aec_task(const char *input_file_name, const char *output_file_name) {
    // Ensure configuration is a subset of the maximum configuration the library supports
    assert(AEC_MAX_Y_CHANNELS <= AEC_LIB_MAX_Y_CHANNELS);
    assert(AEC_MAX_X_CHANNELS <= AEC_LIB_MAX_X_CHANNELS);
    assert((AEC_MAX_Y_CHANNELS * AEC_MAX_X_CHANNELS * AEC_MAIN_FILTER_PHASES) <= (AEC_LIB_MAX_PHASES));
    assert((AEC_MAX_Y_CHANNELS * AEC_MAX_X_CHANNELS * AEC_SHADOW_FILTER_PHASES) <= (AEC_LIB_MAX_PHASES));
    
    file_t input_file, output_file;
    // Open input wav file containing mic and ref channels of input data
    int ret = file_open(&input_file, input_file_name, "rb");
    assert((!ret) && "Failed to open file");
    // Open output wav file that will contain the AEC output
    ret = file_open(&output_file, output_file_name, "wb");
    assert((!ret) && "Failed to open file");

    wav_header input_header_struct, output_header_struct;
    unsigned input_header_size;
    if(get_wav_header_details(&input_file, &input_header_struct, &input_header_size) != 0){
        printf("error in get_wav_header_details()\n");
        _Exit(1);
    }
    file_seek(&input_file, input_header_size, SEEK_SET);
    // Ensure 32bit wav file
    if(input_header_struct.bit_depth != 32)
     {
         printf("Error: unsupported wav bit depth (%d) for %s file. Only 32 supported\n", input_header_struct.bit_depth, input_file_name);
         _Exit(1);
     }
    // Ensure input wav file contains correct number of channels 
    if(input_header_struct.num_channels != (AEC_MAX_Y_CHANNELS+AEC_MAX_X_CHANNELS)){
        printf("Error: wav num channels(%d) does not match aec(%u)\n", input_header_struct.num_channels, (AEC_MAX_Y_CHANNELS+AEC_MAX_X_CHANNELS));
        _Exit(1);
    }
    
    unsigned frame_count = wav_get_num_frames(&input_header_struct);
    // Calculate number of frames in the wav file
    unsigned block_count = frame_count / AEC_FRAME_ADVANCE;
    wav_form_header(&output_header_struct,
            input_header_struct.audio_format,
            AEC_MAX_Y_CHANNELS,
            input_header_struct.sample_rate,
            input_header_struct.bit_depth,
            block_count*AEC_FRAME_ADVANCE);

    file_write(&output_file, (uint8_t*)(&output_header_struct),  WAV_HEADER_BYTES);

    int32_t input_read_buffer[AEC_FRAME_ADVANCE * (AEC_MAX_Y_CHANNELS + AEC_MAX_X_CHANNELS)] = {0}; // Array for storing interleaved input read from wav file
    int32_t output_write_buffer[AEC_FRAME_ADVANCE * (AEC_MAX_Y_CHANNELS)];

    int32_t DWORD_ALIGNED frame_y[AEC_MAX_Y_CHANNELS][AEC_FRAME_ADVANCE];
    int32_t DWORD_ALIGNED frame_x[AEC_MAX_X_CHANNELS][AEC_FRAME_ADVANCE];

    unsigned bytes_per_frame = wav_get_num_bytes_per_frame(&input_header_struct);

    // Initialise AEC
    uint8_t DWORD_ALIGNED aec_memory_pool[sizeof(aec_memory_pool_t)];
    uint8_t DWORD_ALIGNED aec_shadow_filt_memory_pool[sizeof(aec_shadow_filt_memory_pool_t)]; 
    aec_state_t DWORD_ALIGNED main_state;
    aec_state_t DWORD_ALIGNED shadow_state;
    aec_shared_state_t DWORD_ALIGNED aec_shared_state;
    
    aec_init(&main_state, &shadow_state, &aec_shared_state,
            &aec_memory_pool[0], &aec_shadow_filt_memory_pool[0],
            AEC_MAX_Y_CHANNELS, AEC_MAX_X_CHANNELS,
            AEC_MAIN_FILTER_PHASES, AEC_SHADOW_FILTER_PHASES);

    for(unsigned b=0;b<block_count;b++){
        long input_location =  wav_get_frame_start(&input_header_struct, b * AEC_FRAME_ADVANCE, input_header_size);
        file_seek (&input_file, input_location, SEEK_SET);
        file_read (&input_file, (uint8_t*)&input_read_buffer[0], bytes_per_frame* AEC_FRAME_ADVANCE);
        // Deinterleave and copy y and x samples to their respective buffers
        for(unsigned f=0; f<AEC_FRAME_ADVANCE; f++){
            for(unsigned ch=0;ch<AEC_MAX_Y_CHANNELS;ch++){
                unsigned i =(f * (AEC_MAX_Y_CHANNELS+AEC_MAX_X_CHANNELS)) + ch;
                frame_y[ch][f] = input_read_buffer[i];
            }
            for(unsigned ch=0;ch<AEC_MAX_X_CHANNELS;ch++){
                unsigned i =(f * (AEC_MAX_Y_CHANNELS+AEC_MAX_X_CHANNELS)) + AEC_MAX_Y_CHANNELS + ch;
                frame_x[ch][f] = input_read_buffer[i];
            }
        }
        // Call AEC functions to process AEC_FRAME_ADVANCE new samples of data
        /* Reuse mic data memory for main filter output
         * Reuse ref data memory for shadow filter output.
         */
        aec_process_frame_2threads(&main_state, &shadow_state, frame_y, frame_x, frame_y, frame_x);
        
        // Create interleaved output that can be written to wav file
        for (unsigned ch=0;ch<AEC_MAX_Y_CHANNELS;ch++){
            for(unsigned i=0;i<AEC_FRAME_ADVANCE;i++){
                output_write_buffer[i*(AEC_MAX_Y_CHANNELS) + ch] = frame_y[ch][i];
            }
        }

        file_write(&output_file, (uint8_t*)(output_write_buffer), output_header_struct.bit_depth/8 * AEC_FRAME_ADVANCE * AEC_MAX_Y_CHANNELS);
    }
    file_close(&input_file);
    file_close(&output_file);
    shutdown_session();
}


#if X86_BUILD
int main(int argc, char **argv) {
    if(argc < 3) {
        printf("Arguments missing. Expected: <input file name> <output file name>\n");
        assert(0);
    }
    aec_task(argv[1], argv[2]);
    return 0;
}
#endif

// Copyright 2022 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>

#include "aec_api.h"
#include "ic_api.h"
#include "ns_api.h"
#include "pipeline_config.h"
#include "pipeline_state.h"
#include "stage_1.h"

#define VNR_AGC_THRESHOLD (0.5)
#define PRINT_VNR_PREDICTION (0)

void pipeline_tile0_init(pipeline_state_tile0_t *state) {
    memset(state, 0, sizeof(pipeline_state_tile0_t)); 
    
    // Initialise AEC, DE, ADEC stage
    aec_conf_t aec_de_mode_conf, aec_non_de_mode_conf;
#if ALT_ARCH_MODE
    aec_non_de_mode_conf.num_y_channels = 1;
    aec_non_de_mode_conf.num_x_channels = AP_MAX_X_CHANNELS;
    aec_non_de_mode_conf.num_main_filt_phases = 15;
    aec_non_de_mode_conf.num_shadow_filt_phases = AEC_SHADOW_FILTER_PHASES;
#else
    aec_non_de_mode_conf.num_y_channels = AP_MAX_Y_CHANNELS;
    aec_non_de_mode_conf.num_x_channels = AP_MAX_X_CHANNELS;
    aec_non_de_mode_conf.num_main_filt_phases = AEC_MAIN_FILTER_PHASES;
    aec_non_de_mode_conf.num_shadow_filt_phases = AEC_SHADOW_FILTER_PHASES;
#endif

    aec_de_mode_conf.num_y_channels = 1;
    aec_de_mode_conf.num_x_channels = 1;
    aec_de_mode_conf.num_main_filt_phases = 30;
    aec_de_mode_conf.num_shadow_filt_phases = 0;
    
    // Disable ADEC's automatic mode. We only want to estimate and correct for the delay at startup
    adec_config_t adec_conf;
    adec_conf.bypass = 1; // Bypass automatic DE correction
#if DISABLE_INITIAL_DELAY_EST
    adec_conf.force_de_cycle_trigger = 0; // Do not force a DE correction cycle ob startup
#else
    adec_conf.force_de_cycle_trigger = 1; // Force a delay correction cycle, so that delay correction happens once after initialisation. Make sure this is set back to 0 after adec has requested a transition into DE mode once, to stop any further delay correction (automatic or forced) by ADEC
#endif
    stage_1_init(&state->stage_1_state, &aec_de_mode_conf, &aec_non_de_mode_conf, &adec_conf);
}

void pipeline_tile1_init(pipeline_state_tile1_t *state) {
    memset(state, 0, sizeof(pipeline_state_tile1_t)); 
    
    // Initialise IC, VNR
    ic_init(&state->ic_state);

    // Initialise NS
    for(int ch = 0; ch < AP_MAX_Y_CHANNELS; ch++){
        ns_init(&state->ns_state[ch]);
    }
    
    // Initialise AGC
    agc_config_t agc_conf_asr = AGC_PROFILE_ASR;
#if DISABLE_AGC_ADAPT_GAIN
    agc_conf_asr.adapt = 0;
#endif
    
    agc_init(&state->agc_state[0], &agc_conf_asr);
    agc_init(&state->agc_state[1], &agc_conf_asr);

}

void pipeline_process_frame_tile0(pipeline_state_tile0_t *state,
        int32_t (*input_y_data)[AP_FRAME_ADVANCE],
        int32_t (*input_x_data)[AP_FRAME_ADVANCE],
        int32_t (*output_data)[AP_FRAME_ADVANCE],
        pipeline_metadata_t *md_output)
{
    pipeline_metadata_t md;
    memset(&md, 0, sizeof(pipeline_metadata_t));

    /** Stage1 - AEC, DE, ADEC*/
    int32_t stage_1_out[AEC_MAX_Y_CHANNELS][AP_FRAME_ADVANCE];// stage1 will not process the frame in-place since Mic input is needed to overwrite the output in certain cases

#if DISABLE_STAGE_1
    memcpy(&stage_1_out[0][0], &input_y_data[0][0], AEC_MAX_Y_CHANNELS*AP_FRAME_ADVANCE*sizeof(int32_t));
#else
    stage_1_process_frame(&state->stage_1_state, &stage_1_out[0], &md.max_ref_energy, &md.aec_corr_factor[0], &md.ref_active_flag, input_y_data, input_x_data);
    
    if(state->stage_1_state.aec_main_state.shared_state->num_y_channels < AP_MAX_Y_CHANNELS) {
        for(int ch=state->stage_1_state.aec_main_state.shared_state->num_y_channels; ch<AP_MAX_Y_CHANNELS; ch++) {
            md.aec_corr_factor[ch] = md.aec_corr_factor[0];
        }
    }
#endif
    memcpy(&output_data[0][0], &stage_1_out[0][0], AEC_MAX_Y_CHANNELS*AP_FRAME_ADVANCE*sizeof(int32_t));
    memcpy(md_output, &md, sizeof(pipeline_metadata_t));
}

void pipeline_process_frame_tile1(pipeline_state_tile1_t *state, pipeline_metadata_t *md_input,
        int32_t (*input_data)[AP_FRAME_ADVANCE],
        int32_t (*output_data)[AP_FRAME_ADVANCE])
{
    pipeline_metadata_t md;
    memcpy(&md, md_input, sizeof(pipeline_metadata_t));

    /** IC and VNR*/
    int32_t ic_output[AP_MAX_Y_CHANNELS][AP_FRAME_ADVANCE];
#if DISABLE_STAGE_2
    memcpy(&ic_output[0][0], &input_data[0][0], AEC_MAX_Y_CHANNELS*AP_FRAME_ADVANCE*sizeof(int32_t));
#else

#if ALT_ARCH_MODE
    if(md.ref_active_flag) {
        state->ic_state.config_params.bypass = 1;
    }
    else {
        state->ic_state.config_params.bypass = 0;
    }
#endif
    // The ASR channel will be produced by IC filtering
    ic_filter(&state->ic_state, input_data[0], input_data[1], ic_output[0]);

    // VNR
    ic_calc_vnr_pred(&state->ic_state, &state->input_vnr_pred, &state->output_vnr_pred);
#if PRINT_VNR_PREDICTION 
        printf("VNR OUTPUT PRED: %ld %d\n", state->output_vnr_pred.mant, state->output_vnr_pred.exp);
        printf("VNR INPUT PRED: %ld %d\n", state->input_vnr_pred.mant, state->input_vnr_pred.exp);
#endif
    float_s32_t agc_vnr_threshold = f32_to_float_s32(VNR_AGC_THRESHOLD);
    md.vnr_pred_flag = float_s32_gt(state->output_vnr_pred, agc_vnr_threshold);
   
    ic_adapt(&state->ic_state, state->input_vnr_pred);

    // Copy IC output to the other channel
    for(int v = 0; v < AP_FRAME_ADVANCE; v++){
        ic_output[1][v] = ic_output[0][v];
    }
#endif

    /** NS*/
    int32_t ns_output[AP_MAX_Y_CHANNELS][AP_FRAME_ADVANCE];
#if DISABLE_STAGE_3
    memcpy(&ns_output[0][0], &ic_output[0][0], AEC_MAX_Y_CHANNELS*AP_FRAME_ADVANCE*sizeof(int32_t));
#else
    for(int ch = 0; ch < AP_MAX_Y_CHANNELS; ch++){
        ns_process_frame(&state->ns_state[ch], ns_output[ch], ic_output[ch]);
    }
#endif

    /** AGC*/
#if DISABLE_STAGE_4
    memcpy(&output_data[0][0], &ns_output[0][0], AEC_MAX_Y_CHANNELS*AP_FRAME_ADVANCE*sizeof(int32_t));
#else
    agc_meta_data_t agc_md;
    agc_md.aec_ref_power = md.max_ref_energy;
    agc_md.vnr_flag = md.vnr_pred_flag;

    for(int ch=0; ch<AP_MAX_Y_CHANNELS; ch++) {
        agc_md.aec_corr_factor = md.aec_corr_factor[ch];
        agc_process_frame(&state->agc_state[ch], output_data[ch], ns_output[ch], &agc_md);
    }
#endif
}


// Copyright 2022 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include <stdlib.h>

#include "aec_defines.h"
#include "aec_api.h"
#include "adec_api.h"

#include "aec_config.h"
#include "aec_task_distribution.h"

#include "pipeline_state.h"

#if PROFILE_PROCESSING
#include "profile.h"
#endif

extern void aec_process_frame_2threads(
        aec_state_t *main_state,
        aec_state_t *shadow_state,
        int32_t (*output_main)[AEC_FRAME_ADVANCE],
        int32_t (*output_shadow)[AEC_FRAME_ADVANCE],
        const int32_t (*y_data)[AEC_FRAME_ADVANCE],
        const int32_t (*x_data)[AEC_FRAME_ADVANCE]);


static void aec_switch_configuration(pipeline_state_t *state, aec_conf_t *conf)
{
    aec_init(&state->aec_main_state, &state->aec_shadow_state, &state->aec_shared_state,
            &state->aec_main_memory_pool[0], &state->aec_shadow_memory_pool[0],
            conf->num_y_channels, conf->num_x_channels,
            conf->num_main_filt_phases, conf->num_shadow_filt_phases);
}


void pipeline_init(pipeline_state_t *state, aec_conf_t *de_conf, aec_conf_t *non_de_conf, adec_config_t *adec_config) {
    memset(state, 0, sizeof(pipeline_state_t)); 
    prof(0, "start_pipeline_init"); //Start profiling after memset since the pipeline components do the memset as part of their init functions.
    state->delay_estimator_enabled = 0;
    state->adec_requested_delay_samples = 0;
    state->ref_active_threshold =  f64_to_float_s32(pow(10, -60/20.0));

    // Initialise default delay values
    delay_buffer_init(&state->delay_state, 0/*Initialise with 0 delay_samples*/);
    
    memcpy(&state->aec_de_mode_conf, de_conf, sizeof(aec_conf_t));
    memcpy(&state->aec_non_de_mode_conf, non_de_conf, sizeof(aec_conf_t));

    adec_init(&state->adec_state, adec_config);
    aec_switch_configuration(state, &state->aec_non_de_mode_conf);
    prof(1, "end_pipeline_init");
}

static inline void get_delayed_frame(
        int32_t (*input_y_data)[AP_FRAME_ADVANCE],
        int32_t (*input_x_data)[AP_FRAME_ADVANCE],
        delay_buf_state_t *delay_state)
{
    int num_channels = (delay_state->delay_samples) > 0 ? AP_MAX_Y_CHANNELS : AP_MAX_X_CHANNELS;
    if (delay_state->delay_samples >= 0) {/** Requested Mic delay +ve => delay mic*/
        for(int ch=0; ch<num_channels; ch++) {
            for(int i=0; i<AP_FRAME_ADVANCE; i++) {
                get_delayed_sample(delay_state, &input_y_data[ch][i], ch);
            }
        }
    }
    else if (delay_state->delay_samples < 0) {/* Requested Mic delay negative => advance mic which can't be done, so delay reference*/
        for(int ch=0; ch<num_channels; ch++) {
            for(int i=0; i<AP_FRAME_ADVANCE; i++) {
                get_delayed_sample(delay_state, &input_x_data[ch][i], ch);
            }
        }
    }
    return;
}

int framenum = 0;
void pipeline_process_frame(pipeline_state_t *state,
        int32_t (*output_data)[AP_FRAME_ADVANCE],
        int32_t (*input_y_data)[AP_FRAME_ADVANCE],
        int32_t (*input_x_data)[AP_FRAME_ADVANCE]
        )
{
    /** Get delayed frame*/
    prof(2, "start_get_delayed_frame");
    delay_buf_state_t *delay_state_ptr = &state->delay_state;
    get_delayed_frame(
            input_y_data,
            input_x_data,
            delay_state_ptr
            );
    prof(3, "end_get_delayed_frame");

    prof(18, "start_switch_aec_config");
    /** Switch AEC config if needed. We're doing it at the beginning of current frame instead of end of previous
     * frame to be able to read relevant outputs from the state in the test_wav code for logging. Switching at the 
     * end of last frame would have reset the state.*/
    if (state->adec_output_delay_estimator_enabled_flag && !state->delay_estimator_enabled) {
        /** Now that a AEC -> DE change has been requested, reset force_de_cycle_trigger in case this transition is
         * requested as a result of force_de_cycle_trigger being set*/ 
        state->adec_state.adec_config.force_de_cycle_trigger = 0;

        // Initialise AEC for delay estimation config
        aec_switch_configuration(state, &state->aec_de_mode_conf);
        state->aec_main_state.shared_state->config_params.coh_mu_conf.adaption_config = AEC_ADAPTION_FORCE_ON;
        //state->aec_main_state.shared_state->config_params.coh_mu_conf.force_adaption_mu_q30 = fixed_mu_delay_est_mode;
        state->delay_estimator_enabled = 1;
    } else if ((!state->adec_output_delay_estimator_enabled_flag && state->delay_estimator_enabled)) {
        // Start AEC for normal aec config
        aec_switch_configuration(state, &state->aec_non_de_mode_conf);
        state->delay_estimator_enabled = 0;
    }
    prof(19, "end_switch_aec_config");
    
    prof(4, "start_is_frame_active");
    // Calculate far_end active
    int is_ref_active = aec_detect_input_activity(input_x_data, state->ref_active_threshold, state->aec_main_state.shared_state->num_x_channels);
    prof(5, "end_is_frame_active");

    //printf("frame %d\n",framenum);

    /** AEC*/
    prof(6, "start_aec_process_frame");
    int32_t aec_output_shadow[AP_MAX_Y_CHANNELS][AP_FRAME_ADVANCE];
    // Writing main filter output to output_data directly

    aec_process_frame_2threads(&state->aec_main_state, &state->aec_shadow_state, output_data, aec_output_shadow, input_y_data, input_x_data);

    prof(7, "end_aec_process_frame");
    
    prof(8, "start_estimate_delay");
    /** Delay estimator*/
    adec_input_t adec_in;
    adec_estimate_delay(
            &adec_in.from_de,
            state->aec_main_state.H_hat[0],
            state->aec_main_state.num_phases
            );
    
    prof(9, "end_estimate_delay");

    prof(10, "start_adec_process_frame");
    /** ADEC*/
    // Create input to ADEC from AEC
    adec_in.from_aec.y_ema_energy_ch0 = state->aec_main_state.shared_state->y_ema_energy[0];
    adec_in.from_aec.error_ema_energy_ch0 = state->aec_main_state.error_ema_energy[0];
    adec_in.from_aec.shadow_flag_ch0 = state->aec_main_state.shared_state->shadow_filter_params.shadow_flag[0];
    // Directly from app
    adec_in.far_end_active_flag = is_ref_active; 
    
    // Log current mode for printing later
#ifdef ENABLE_ADEC_DEBUG_PRINTS
    adec_mode_t old_mode = state->adec_state.mode;
#endif
    
    // Call ADEC
    adec_output_t adec_output;
    adec_process_frame(
            &state->adec_state,
            &adec_output,
            &adec_in
            );
    
    prof(11, "end_adec_process_frame");

    prof(12, "start_reset_aec");
    //** Reset AEC state if needed*/
    if(adec_output.reset_aec_flag) {
        aec_reset_state(&state->aec_main_state, &state->aec_shadow_state);
    }
    prof(13, "end_reset_aec");
    
    prof(14, "start_update_delay_buffer");
    /** Update delay samples if there's a delay change requested by ADEC*/
    if(adec_output.delay_change_request_flag == 1){
        // Update delay_buffer delay_samples with mic delay requested by adec
        update_delay_samples(&state->delay_state, adec_output.requested_mic_delay_samples);
        for(int ch=0; ch<AP_MAX_Y_CHANNELS; ch++) {
            reset_partial_delay_buffer(&state->delay_state, ch);
        }
#ifdef ENABLE_ADEC_DEBUG_PRINTS
        printf("!!ADEC STATE CHANGE!! Frame: %d old: %s new: %s\n", framenum, old_mode?"DE":"AEC", state->adec_state.mode?"DE":"AEC");

        printf("AP Setting MIC delay to: %ld\n", state->delay_state.delay_samples);
#endif
    }
    prof(15, "end_update_delay_buffer");
    
    prof(16, "start_overwrite_output_in_de_mode");
    /** Overwrite output with mic input if delay estimation enabled*/
    if (state->delay_estimator_enabled) {
        // Send the current frame unprocessed
        for(int ch=0; ch<AP_MAX_Y_CHANNELS; ch++) {
            for(int i=0; i<AP_FRAME_ADVANCE; i++) {
                output_data[ch][i] = input_y_data[ch][i];
            }
        }
    }
    state->adec_output_delay_estimator_enabled_flag = adec_output.delay_estimator_enabled_flag; //Keep a copy so we can use it in next frame
    state->de_output_measured_delay_samples = adec_in.from_de.measured_delay_samples;
    prof(17, "end_overwrite_output_in_de_mode");


    //For logging for test purposes
    /* If a delay change is requested and this request is not accompanied by a AEC -> DE change, update
     * state->adec_requested_delay_samples(bits [0:30]) and toggle last bit (bit31) of
     * state->adec_requested_delay_samples to indicate to the test that a delay change event has occured*/
    if((adec_output.delay_change_request_flag == 1) && (adec_output.delay_estimator_enabled_flag == 0)){
        /* Update MSB 31 bits of state->adec_requested_delay_samples with the new delay. Toggle LSB bit of
         * state->adec_requested_delay_samples to indicate that a new delay (even if it happens to be same as the
         * existing delay) is requested by adec*/
         unsigned new_bottom_bit = (state->adec_requested_delay_samples & 0x1) ^ 0x1; 
         state->adec_requested_delay_samples = adec_output.requested_delay_samples_debug;
         state->adec_requested_delay_samples = (state->adec_requested_delay_samples & 0xfffffffe) | new_bottom_bit;
    }

    framenum++;
    print_prof(0, 20, framenum);
}

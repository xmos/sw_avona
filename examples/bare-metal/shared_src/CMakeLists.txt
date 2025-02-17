
add_subdirectory(external_deps)

######
# add_library(fwk_voice_example_shared_src_aec_1_thread INTERFACE)
# target_sources(fwk_voice_example_shared_src_aec_1_thread
#     INTERFACE
#         aec/aec_process_frame_1thread.c
# )
# target_include_directories(fwk_voice_example_shared_src_aec_1_thread
#     INTERFACE
#         aec
# )
# target_link_libraries(fwk_voice_example_shared_src_aec_1_thread
#     INTERFACE
#         fwk_voice::aec
# )
# add_library(fwk_voice::example::aec1thread ALIAS fwk_voice_example_shared_src_aec_1_thread)


######
add_library(fwk_voice_example_shared_src_aec_2_thread INTERFACE)
target_sources(fwk_voice_example_shared_src_aec_2_thread
    INTERFACE
        aec/aec_process_frame_2threads.c
)
target_include_directories(fwk_voice_example_shared_src_aec_2_thread
    INTERFACE
        aec
)
target_link_libraries(fwk_voice_example_shared_src_aec_2_thread
    INTERFACE
        fwk_voice::aec
)
add_library(fwk_voice::example::aec2thread ALIAS fwk_voice_example_shared_src_aec_2_thread)

######
add_library(fwk_voice_example_shared_src_delay_buffer  INTERFACE)
target_sources(fwk_voice_example_shared_src_delay_buffer
    INTERFACE
        delay_buffer/delay_buffer.c
)
target_include_directories(fwk_voice_example_shared_src_delay_buffer
    INTERFACE
        delay_buffer
)
add_library(fwk_voice::example::delay_buffer ALIAS fwk_voice_example_shared_src_delay_buffer)

######
add_library(fwk_voice_example_shared_src_stage_1  INTERFACE)
target_sources(fwk_voice_example_shared_src_stage_1
    INTERFACE
        pipeline_stage_1/stage_1.c
)
target_include_directories(fwk_voice_example_shared_src_stage_1
    INTERFACE
        pipeline_stage_1
)
add_library(fwk_voice::example::stage_1 ALIAS fwk_voice_example_shared_src_stage_1)

######
add_library(fwk_voice_example_shared_src_file_utils INTERFACE)
target_sources(fwk_voice_example_shared_src_file_utils
    INTERFACE
        file_utils/fileio.c
        file_utils/wav_utils.c
)
target_include_directories(fwk_voice_example_shared_src_file_utils
    INTERFACE
        file_utils
)
if(${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A)
    target_link_libraries(fwk_voice_example_shared_src_file_utils
        INTERFACE
            fwk_voice_deps::xscope_fileio)
endif()

add_library(fwk_voice::example::fileutils ALIAS fwk_voice_example_shared_src_file_utils)

######
add_library(fwk_voice_example_shared_src_profile_xcore INTERFACE)
target_sources(fwk_voice_example_shared_src_profile_xcore
    INTERFACE
        profile/profile.c
)
target_include_directories(fwk_voice_example_shared_src_profile_xcore
    INTERFACE
        profile
)

add_library(fwk_voice::example::profile_xcore ALIAS fwk_voice_example_shared_src_profile_xcore)

######
add_library(fwk_voice_example_shared_src_hpf INTERFACE)
target_sources(fwk_voice_example_shared_src_hpf
    INTERFACE
        hpf/hpf.c
)
target_include_directories(fwk_voice_example_shared_src_hpf
    INTERFACE
        hpf
)
target_link_libraries(fwk_voice_example_shared_src_hpf
    INTERFACE
        lib_xcore_math
)

add_library(fwk_voice::example::hpf ALIAS fwk_voice_example_shared_src_hpf)

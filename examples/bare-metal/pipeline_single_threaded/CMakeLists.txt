
### Add interface library
add_library(fwk_voice_example_pipeline_single_thread_interface INTERFACE)

file( GLOB_RECURSE SOURCES_APP  src/*.c )
target_sources(fwk_voice_example_pipeline_single_thread_interface
    INTERFACE
        ${SOURCES_APP})

target_include_directories(fwk_voice_example_pipeline_single_thread_interface
    INTERFACE
        src)

target_link_libraries(fwk_voice_example_pipeline_single_thread_interface
    INTERFACE
        fwk_voice::aec
        fwk_voice::adec
        fwk_voice::ns
        fwk_voice::agc
        fwk_voice::ic
        fwk_voice::example::aec2thread
        fwk_voice::example::delay_buffer
        fwk_voice::example::stage_1
        fwk_voice::example::fileutils
        lib_xcore_math
        )

if(${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A)
    target_sources(fwk_voice_example_pipeline_single_thread_interface
        INTERFACE src/main.xc)

    target_compile_options(fwk_voice_example_pipeline_single_thread_interface
        INTERFACE "-target=${XCORE_TARGET}")

endif()

### Add executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(fwk_voice_example_bare_metal_pipeline_single_thread)

target_link_libraries(fwk_voice_example_bare_metal_pipeline_single_thread
    PRIVATE
        fwk_voice_example_pipeline_single_thread_interface
        )

if(${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A)
    target_link_options(fwk_voice_example_bare_metal_pipeline_single_thread
        PRIVATE
            "-target=${XCORE_TARGET}"
            "-report"
            "${CONFIG_XSCOPE_PATH}/config.xscope")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    # Static linking doesn't work on Mac(ld: library not found for -lcrt0.o)
    target_link_libraries(fwk_voice_example_bare_metal_pipeline_single_thread
        PRIVATE m)
else()
    # Note we link this app statically so that we can run it on a different machine
    # that may not have all of the dependancies in the right places
    target_link_libraries(fwk_voice_example_bare_metal_pipeline_single_thread
        PRIVATE m "-static")
endif()

### Build AEC + IC pipeline for testing against python AEC+IC pipeline

add_executable(fwk_voice_example_st_pipeline_aec_ic)

target_link_libraries(fwk_voice_example_st_pipeline_aec_ic
    PRIVATE
        fwk_voice_example_pipeline_single_thread_interface
        )
target_compile_definitions(fwk_voice_example_st_pipeline_aec_ic
    PRIVATE
        DISABLE_STAGE_3=1
        DISABLE_INITIAL_DELAY_EST=1
        DISABLE_AGC_ADAPT_GAIN=1
    )

if(${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A)
    target_link_options(fwk_voice_example_st_pipeline_aec_ic
        PRIVATE
            "-target=${XCORE_TARGET}"
            "-report"
            "${CONFIG_XSCOPE_PATH}/config.xscope")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    # Static linking doesn't work on Mac(ld: library not found for -lcrt0.o)
    target_link_libraries(fwk_voice_example_st_pipeline_aec_ic
        PRIVATE m)
else()
    # Note we link this app statically so that we can run it on a different machine
    # that may not have all of the dependancies in the right places
    target_link_libraries(fwk_voice_example_st_pipeline_aec_ic
        PRIVATE m "-static")
endif()

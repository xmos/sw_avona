
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(fwk_voice_example_bare_metal_aec_1_thread)

target_sources(fwk_voice_example_bare_metal_aec_1_thread
    PRIVATE
        src/aec_task_distribution.c
        src/aec_app.c
)

target_include_directories(fwk_voice_example_bare_metal_aec_1_thread
    PRIVATE
        src
)

target_link_libraries(fwk_voice_example_bare_metal_aec_1_thread
    PUBLIC
        fwk_voice::example::aec2thread
        lib_xcore_math
        fwk_voice::example::fileutils
)

if(${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A)
    target_sources(fwk_voice_example_bare_metal_aec_1_thread
        PRIVATE
            src/main.xc
    )
    target_compile_options(fwk_voice_example_bare_metal_aec_1_thread
        PRIVATE
            "-target=${XCORE_TARGET}"
    )
    target_link_options(fwk_voice_example_bare_metal_aec_1_thread
        PRIVATE
            "-target=${XCORE_TARGET}"
            "-report"
            "${CONFIG_XSCOPE_PATH}/config.xscope"
    )
else()
    target_link_libraries(fwk_voice_example_bare_metal_aec_1_thread
        PRIVATE
            m
    )
endif()

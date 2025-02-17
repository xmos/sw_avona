set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

if( NOT ${Python3_FOUND} )
  message(FATAL_ERROR "Python3 not found for running . ") 
endif()

set( GEN_SCHEDULE_SCRIPT ${SHARED_SRC_PATH}/python/generate_task_distribution_scheme.py )
set( AUTOGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/src.autogen )
set( AUTOGEN_SOURCES ${AUTOGEN_DIR}/aec_task_distribution.c )
set( AUTOGEN_INCLUDES ${AUTOGEN_DIR}/aec_task_distribution.h ${AUTOGEN_DIR}/aec_config.h)

message(STATUS "fwk_voice_test_wav_adec build config:  ${TEST_WAV_ADEC_BUILD_CONFIG}" )
set( GEN_SCHEDULE_SCRIPT_BYPRODUCTS ${AUTOGEN_SOURCES} ${AUTOGEN_INCLUDES} )

unset(GEN_SCHEDULE_SCRIPT_ARGS) 
list(APPEND GEN_SCHEDULE_SCRIPT_ARGS --out-dir ${AUTOGEN_DIR})
list(APPEND GEN_SCHEDULE_SCRIPT_ARGS --config ${TEST_WAV_ADEC_BUILD_CONFIG})

file(MAKE_DIRECTORY ${AUTOGEN_DIR})

add_custom_command(
OUTPUT ${GEN_SCHEDULE_SCRIPT_BYPRODUCTS}
COMMAND ${Python3_EXECUTABLE} ${GEN_SCHEDULE_SCRIPT} ${GEN_SCHEDULE_SCRIPT_ARGS}
COMMENT "Generating fwk_voice_test_wav_adec task distribution and top level config" )

######################
file(GLOB SOURCES_TEST_WAV src/*.c)
add_library(fwk_voice_test_shared_test_wav_adec INTERFACE)
target_sources(fwk_voice_test_shared_test_wav_adec
    INTERFACE
        ${SOURCES_TEST_WAV})
target_include_directories(fwk_voice_test_shared_test_wav_adec
    INTERFACE
        src)
target_link_libraries(fwk_voice_test_shared_test_wav_adec
    INTERFACE
        fwk_voice::aec
        fwk_voice::adec
        fwk_voice::example::aec2thread
        fwk_voice::example::delay_buffer
        fwk_voice::example::fileutils)

if(${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A)
    target_sources(fwk_voice_test_shared_test_wav_adec
        INTERFACE
            src/main.xc)
    target_link_libraries(fwk_voice_test_shared_test_wav_adec
        INTERFACE
            fwk_voice::example::aec2thread)
endif()

add_library(fwk_voice::test::shared_test_wav_adec ALIAS fwk_voice_test_shared_test_wav_adec)

####################
add_executable(fwk_voice_test_wav_adec)

target_sources(fwk_voice_test_wav_adec
    PRIVATE
        ${AUTOGEN_SOURCES})

target_include_directories(fwk_voice_test_wav_adec
    PRIVATE
        ${AUTOGEN_DIR})

target_link_libraries(fwk_voice_test_wav_adec
    PUBLIC
        fwk_voice::test::shared_test_wav_adec)

target_compile_definitions(fwk_voice_test_wav_adec
    PRIVATE
        ENABLE_ADEC_DEBUG_PRINTS)

if(${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A)
    target_compile_options(fwk_voice_test_wav_adec
        PRIVATE "-target=${XCORE_TARGET}")

    target_link_options(fwk_voice_test_wav_adec
        PRIVATE
            "-target=${XCORE_TARGET}"
            "-report"
            "${CMAKE_CURRENT_SOURCE_DIR}/config.xscope")
else()
    target_link_libraries(fwk_voice_test_wav_adec
        PRIVATE m)
endif()
    
    


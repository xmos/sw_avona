
# Auto-generate task distribution scheme and top level config files

if( NOT ${Python3_FOUND} )
  message(FATAL_ERROR "Python3 not found for running . ") 
endif()

#copy conftest.py in the build directory since pytest_collect_file only collects tests from the directory tree where conftest.py is present
configure_file( conftest.py conftest.py COPYONLY ) 
set( GEN_SCHEDULE_SCRIPT ${SHARED_SRC_PATH}/python/generate_task_distribution_scheme.py )
set( AUTOGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/src.autogen )
set( AUTOGEN_SOURCES ${AUTOGEN_DIR}/aec_task_distribution.c )
set( AUTOGEN_INCLUDES ${AUTOGEN_DIR}/aec_task_distribution.h ${AUTOGEN_DIR}/aec_config.h)

message(STATUS "aec_unit_tests aec build config:  ${AEC_UNIT_TESTS_BUILD_CONFIG}" )
message( STATUS "Unit tests speedup factor ${TEST_SPEEDUP_FACTOR}" )
set( GEN_SCHEDULE_SCRIPT_BYPRODUCTS ${AUTOGEN_SOURCES} ${AUTOGEN_INCLUDES} )

unset(GEN_SCHEDULE_SCRIPT_ARGS) 
list(APPEND GEN_SCHEDULE_SCRIPT_ARGS --out-dir ${AUTOGEN_DIR})
list(APPEND GEN_SCHEDULE_SCRIPT_ARGS --config ${AEC_UNIT_TESTS_BUILD_CONFIG})

file(MAKE_DIRECTORY ${AUTOGEN_DIR})

add_custom_command(
OUTPUT ${GEN_SCHEDULE_SCRIPT_BYPRODUCTS}
COMMAND ${Python3_EXECUTABLE} ${GEN_SCHEDULE_SCRIPT} ${GEN_SCHEDULE_SCRIPT_ARGS}
COMMENT "Generating AEC task distribution and top level config" )

## executable output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Set unity runner generate script
set( GEN_RUNNER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/generate_unity_runner.py )

# Create directory for runner files
set( RUNNERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/src.runners )
file( MAKE_DIRECTORY ${RUNNERS_DIR} )

# Create directory for runner files
set( RUNNERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/src.runners )
file( MAKE_DIRECTORY ${RUNNERS_DIR} )

## Sources
file( GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c )

# For every source file in aec_unit_tests/src
foreach( testfile ${TEST_SOURCES} )
    # Get test name
    #cmake_path( GET testfile STEM TESTNAME )
    get_filename_component(TESTNAME ${testfile} NAME_WLE)
    
    # Create runner file directory
    file( MAKE_DIRECTORY ${RUNNERS_DIR}/${TESTNAME} )

    #########
    ## Create runner file
    set( RUNNER_FILE ${RUNNERS_DIR}/${TESTNAME}/${TESTNAME}_Runner.c )
    set( GEN_RUNNER_SCRIPT_BYPRODUCTS ${RUNNER_FILE} )

    unset(GEN_RUNNER_SCRIPT_ARGS) 
    list(APPEND GEN_RUNNER_SCRIPT_ARGS --project-root ${UNITY_PATH}/.. )
    list(APPEND GEN_RUNNER_SCRIPT_ARGS --source-file ${testfile} )
    list(APPEND GEN_RUNNER_SCRIPT_ARGS --runner-file ${RUNNER_FILE})

    ## Add command to generate runner file
    add_custom_command(
        OUTPUT ${RUNNER_FILE}
        COMMAND ${Python3_EXECUTABLE} ${GEN_RUNNER_SCRIPT} ${GEN_RUNNER_SCRIPT_ARGS}
        COMMENT "Generating AEC runner" )

    #########
    ## Add a build target
    add_executable(fwk_voice_${TESTNAME})

    target_sources(fwk_voice_${TESTNAME}
        PRIVATE
            ${testfile}
            ${AUTOGEN_SOURCES}
            ${RUNNER_FILE})


    target_include_directories(fwk_voice_${TESTNAME}
        PRIVATE
            src
            ${AUTOGEN_DIR}
        )

    target_link_libraries(fwk_voice_${TESTNAME} 
        PUBLIC
            fwk_voice::aec
            fwk_voice::adec
            fwk_voice::test::shared::test_utils
            fwk_voice::test::shared::unity
            fwk_voice::example::aec2thread)

    if(${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A)
        target_compile_options(fwk_voice_${TESTNAME}
            PRIVATE 
                "-DUNITY_SUPPORT_64"
                "-Wno-xcore-fptrgroup"
                "-report"
                "-DSPEEDUP_FACTOR=${TEST_SPEEDUP_FACTOR}")

        target_link_options(fwk_voice_${TESTNAME}
            PRIVATE
                "-target=${XCORE_TARGET}")
    else()
        target_link_libraries(fwk_voice_${TESTNAME} m)
    endif()
endforeach( testfile ${TEST_SOURCES} )


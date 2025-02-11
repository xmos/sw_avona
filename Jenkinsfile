@Library('xmos_jenkins_shared_library@v0.38.0') _

def runningOn(machine) {
  println "Stage running on:"
  println machine
}

getApproval()

pipeline {
  agent none

  parameters {
    string(
      name: 'TOOLS_VERSION',
      defaultValue: '15.3.0',
      description: 'The XTC tools version'
    )
    string(
      name: 'XMOSDOC_VERSION',
      defaultValue: 'v6.3.1',
      description: 'The xmosdoc version'
    )
    booleanParam(name: 'FULL_TEST_OVERRIDE',
                 defaultValue: false,
                 description: 'Force a full test. This increases the number of iterations/scope in some tests')
    booleanParam(name: 'PIPELINE_FULL_RUN',
                 defaultValue: false,
                 description: 'Enables pipelines characterisation test which takes 5.0hrs by itself. Normally run nightly')
  }
  environment {
    REPO = 'fwk_voice'
    FULL_TEST = """${(params.FULL_TEST_OVERRIDE
                    || env.BRANCH_NAME == 'develop'
                    || env.BRANCH_NAME == 'main'
                    || env.BRANCH_NAME ==~ 'release/.*') ? 1 : 0}"""
    PIPELINE_FULL_RUN = """${params.PIPELINE_FULL_RUN ? 1 : 0}"""
  }
  options {
    skipDefaultCheckout()
    timestamps()
    buildDiscarder(xmosDiscardBuildSettings(onlyArtifacts=false))
  }
  stages {
    stage('Build and Docs') {
      parallel {
        stage('Build Docs') {
          agent {
            label "documentation"
          }
          steps {
            checkout scm
            warnError("Docs") {
              buildDocs(archiveZipOnly: true)
            }
          }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        }
        stage('xcore.ai executables build') {
          when {
            expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }
          }
          agent {
            label 'x86_64&&linux'
          }
          stages {
            stage('Get view') {
              steps {
                runningOn(env.NODE_NAME)

                sh "git clone --depth 1 --branch v2.5.2 git@github.com:ThrowTheSwitch/Unity.git"

                dir("${REPO}") {
                  checkout scm
                  sh "git submodule update --init --recursive --jobs 4"

                  // need ai_tools for the build
                  // need numpy to generate aec tests, will get in from ai_tools
                  createVenv(reqFile: "requirements.txt")
                }
              }
            }
            stage('CMake') {
              steps {
                // Do xcore files
                dir("${REPO}/build") {
                  withTools(params.TOOLS_VERSION) {
                    withVenv {
                      script {
                          if (env.FULL_TEST == "1") {
                            sh 'cmake -S.. --toolchain=../xmos_cmake_toolchain/xs3a.cmake -DFWK_VOICE_BUILD_TESTS=ON'
                          }
                          else {
                            sh 'cmake -S.. --toolchain=../xmos_cmake_toolchain/xs3a.cmake -DTEST_SPEEDUP_FACTOR=4 -DFWK_VOICE_BUILD_TESTS=ON'
                          }
                      }
                      sh 'make -j$(nproc)'
                    }
                  }
                }
                dir("${REPO}") {
                  // Stash all executables and xscope_fileio
                  stash name: 'cmake_build_xcore', includes: 'build/**/*.xe, build/**/conftest.py, build/**/xscope_fileio/**'
                }
              }
            }
          }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        }
      }
    }
    stage('xcore.ai Verification') {
      when {
        expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }
      }
      agent {
        label 'xcore.ai'
      }
      stages{
        stage('Get View') {
          steps {
            runningOn(env.NODE_NAME)

            sh "git clone --depth 1 --branch v2.5.2 git@github.com:ThrowTheSwitch/Unity.git"
            sh "git clone --depth 1 --branch v2.0.0 git@github0.xmos.com:xmos-int/xtagctl.git"
            sh "git clone --depth 1 --branch new_pinned_versions git@github.com:xmos/audio_test_tools.git"
            sh "git clone --depth 1 --branch develop git@github.com:xmos/py_voice.git"
            sh "git clone --depth 1 --branch main git@github.com:xmos/amazon_wwe.git"
            sh "git clone --depth 1 --branch master git@github.com:xmos/sensory_sdk.git"

            dir("${REPO}") {
              checkout scm
              sh "git submodule update --init --recursive --jobs 4"

              createVenv(reqFile: "requirements_test.txt")
              withVenv {
                // Note xscope_fileio is fetched by build so install in next stage
                sh "pip install -e ${env.WORKSPACE}/xtagctl"
              }
            }
          }
        }
        stage('Make/get bins and libs'){
          steps {
            dir("${REPO}") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  // Build x86 versions locally as we had problems with moving bins and libs over from previous build due to brew
                  dir("build") {
                    sh "cmake --version"
                    sh 'cmake -S.. -DTEST_WAV_ADEC_BUILD_CONFIG="1 2 2 10 5" -DFWK_VOICE_BUILD_TESTS=ON'
                    sh 'make -j$(nproc)'

                    // We need to put this here because it is not fetched until we build
                    sh "pip install -e fwk_voice_deps/xscope_fileio"
                  }
                  // We do this again on the NUCs for verification later, but this just checks we have no build error
                  dir("test/lib_ic/py_c_frame_compare") {
                    sh "python build_ic_frame_proc.py"
                  }
                  // We do this again on the NUCs for verification later, but this just checks we have no build error
                  dir("test/lib_vnr/py_c_feature_compare") {
                    sh "python build_vnr_feature_extraction.py"
                  }
                  dir("test/stage_b") {
                    sh "python build_c_code.py"
                  }
                  unstash 'cmake_build_xcore'
                }
              }
            }
          }
        }
        stage('Reset XTAGs'){
          steps{
            dir("${REPO}") {
              sh 'rm -f ~/.xtag/acquired' // Hacky but ensure it always works even when previous failed run left lock file present
              withTools(params.TOOLS_VERSION) {
                withVenv{
                  sh "xtagctl reset_all XCORE-AI-EXPLORER"
                }
              }
            }
          }
        }
        stage('Examples') {
          steps {
            dir("${REPO}") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  dir("examples/bare-metal/aec_1_thread") {
                    sh "python ../shared_src/python/run_xcoreai.py ../../../build/examples/bare-metal/aec_1_thread/bin/fwk_voice_example_bare_metal_aec_1_thread.xe --input ../shared_src/test_streams/aec_example_input.wav"
                  }
                  dir("examples/bare-metal/aec_2_threads") {
                    sh "python ../shared_src/python/run_xcoreai.py ../../../build/examples/bare-metal/aec_2_threads/bin/fwk_voice_example_bare_metal_aec_2_thread.xe --input ../shared_src/test_streams/aec_example_input.wav"
                    // Make sure 1 thread and 2 threads output is bitexact
                    sh "diff output.wav ../aec_1_thread/output.wav"
                  }
                  dir("examples/bare-metal/ic") {
                    sh "python ../shared_src/python/run_xcoreai.py ../../../build/examples/bare-metal/ic/bin/fwk_voice_example_bare_metal_ic.xe"
                    sh "mv output.wav ic_example_output.wav"
                  }
                  dir("examples/bare-metal/pipeline_single_threaded") {
                    sh "python ../shared_src/python/run_xcoreai.py ../../../build/examples/bare-metal/pipeline_single_threaded/bin/fwk_voice_example_bare_metal_pipeline_single_thread.xe --input ../shared_src/test_streams/pipeline_example_input.wav"
                  }
                  dir("examples/bare-metal/pipeline_multi_threaded") {
                    sh "python ../shared_src/python/run_xcoreai.py ../../../build/examples/bare-metal/pipeline_multi_threaded/bin/fwk_voice_example_bare_metal_pipeline_multi_thread.xe --input ../shared_src/test_streams/pipeline_example_input.wav"
                    // Make sure single thread and multi threads pipeline output is bitexact
                    sh "diff output.wav ../pipeline_single_threaded/output.wav"
                  }
                  dir("examples/bare-metal/pipeline_alt_arch") {
                    sh "python ../shared_src/python/run_xcoreai.py ../../../build/examples/bare-metal/pipeline_alt_arch/bin/fwk_voice_example_bare_metal_pipeline_alt_arch_st.xe --input ../shared_src/test_streams/pipeline_example_input.wav"
                    sh "mv output.wav output_st.wav"

                    sh "python ../shared_src/python/run_xcoreai.py ../../../build/examples/bare-metal/pipeline_alt_arch/bin/fwk_voice_example_bare_metal_pipeline_alt_arch_mt.xe --input ../shared_src/test_streams/pipeline_example_input.wav"
                    sh "mv output.wav output_mt.wav"
                    sh "diff output_st.wav output_mt.wav"
                  }
                  dir("examples/bare-metal/agc") {
                    sh "python ../shared_src/python/run_xcoreai.py ../../../build/examples/bare-metal/agc/bin/fwk_voice_example_bare_metal_agc.xe --input ../shared_src/test_streams/agc_example_input.wav"
                  }
                  dir("examples/bare-metal/vnr") {
                    sh "python host_app.py test_stream_1.wav vnr_out2.bin --run-with-xscope-fileio" // With xscope host in lib xscope_fileio
                    sh "python host_app.py test_stream_1.wav vnr_out1.bin" // With xscope host in python
                    sh "diff vnr_out1.bin vnr_out2.bin"
                  }
                }
              }
            }
          }
        }

        stage('VNR tests') {
          steps {
            dir("${REPO}/test/lib_vnr") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  withEnv(["hydra_audio_PATH=/projects/hydra_audio"]) {
                    dir("test_wav_vnr") {
                      sh "pytest -n 1 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("vnr_unit_tests") {
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("py_c_feature_compare") {
                      sh "python build_vnr_feature_extraction.py"
                      sh "pytest -s --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                  }
                }
              }
            }
          }
        }

        stage('NS tests') {
          steps {
            dir("${REPO}/test/lib_ns") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  withEnv(["hydra_audio_PATH=/projects/hydra_audio"]) {
                    dir("test_ns_profile"){
                      sh "pytest -n 1 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("compare_c_py"){
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("ns_unit_tests"){
                      sh "pytest -n 1 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                  }
                }
              }
            }
          }
        }

        stage('IC tests') {
          steps {
            dir("${REPO}/test/lib_ic") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  withEnv(["hydra_audio_PATH=/projects/hydra_audio"]) {
                    dir("ic_unit_tests"){
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("py_c_frame_compare"){
                      sh "python build_ic_frame_proc.py"
                      sh "pytest -s --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("test_ic_profile"){
                      sh "pytest --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("test_ic_spec"){
                      // This test compares the model and C implementation over a range of scenarious for:
                      // convergence_time, db_suppression, maximum noise added to input (to test for stability)
                      // and expected group delay. It will fail if these are not met.
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                      sh "python print_stats.py > ic_spec_summary.txt"
                      // This script generates a number of polar plots of attenuation vs null point angle vs freq
                      // It currently only uses the python model to do this. It takes about 40 mins for all plots
                      // and generates a series of IC_performance_xxxHz.svg files which could be archived
                      //sh "python plot_ic.py"
                    }
                    dir("characterise_c_py"){
                      // This test compares the suppression performance across angles between model and C implementation
                      // and fails if they differ significantly. It requires that the C implementation run with fixed mu
                      sh "pytest -s --junitxml=pytest_result.xml" // -n 2 fails often so run single threaded and also print result
                      junit "pytest_result.xml"
                      // This script sweeps the y_delay value to find what the optimum suppression is across RT60 and angle.
                      // It's more of a model develpment tool than testing the implementation so not run. It take a few minutes.
                      //sh "python sweep_ic_delay.py"
                    }
                    dir("test_calc_vnr_pred"){
                      // This is a unit test for ic_calc_vnr_pred function.
                      sh "pytest -n1 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("test_bad_state"){
                      sh "pytest -s --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                  }
                }
              }
            }
          }
        }

        stage('Stage B tests') {
          steps {
            dir("${REPO}/test/stage_b") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  withEnv(["hydra_audio_PATH=/projects/hydra_audio"]) {
                    sh "pytest -n 1 --junitxml=pytest_result.xml"
                    junit "pytest_result.xml"
                  }
                }
              }
            }
          }
        }

        stage('ADEC tests') {
          steps {
            dir("${REPO}/test/lib_adec") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  withEnv(["hydra_audio_PATH=/projects/hydra_audio"]) {
                    dir("de_unit_tests") {
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("test_delay_estimator") {
                      sh 'mkdir -p ./input_wavs/'
                      sh 'mkdir -p ./output_files/'
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                      sh "python print_stats.py"
                    }
                    dir("test_adec_startup") {
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("test_adec") {
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("test_adec_profile") {
                      sh "pytest -n 1 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                  }
                }
              }
            }
          }
        }

        stage('AEC tests') {
          steps {
            dir("${REPO}/test/lib_aec") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  withEnv(["hydra_audio_PATH=/projects/hydra_audio"]) {
                    dir("test_aec_enhancements") {
                      sh "./make_dirs.sh"
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("aec_unit_tests") {
                      sh "pytest -n 2 --junitxml=pytest_result.xml"
                      junit "pytest_result.xml"
                    }
                    dir("test_aec_spec") {
                      sh "./make_dirs.sh"
                      script {
                        if (env.FULL_TEST == "0") {
                          sh 'mv excluded_tests_quick.txt excluded_tests.txt'
                        }
                      }
                      sh "python generate_audio.py"
                      sh "pytest -n 2 --junitxml=pytest_result.xml test_process_audio.py"
                      sh "cp pytest_result.xml results_process.xml"
                      catchError {
                        sh "pytest --junitxml=pytest_result.xml test_check_output.py"
                      }
                      sh "cp pytest_result.xml results_check.xml"
                      sh "python parse_results.py"
                      sh "pytest --junitxml=pytest_results.xml test_evaluate_results.py"
                      sh "cp pytest_result.xml results_final.xml"
                      junit "results_final.xml"
                    }
                  }
                }
              }
            }
          }
        }

        stage('AGC tests') {
          steps {
            dir("${REPO}/test/lib_agc/test_process_frame") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  sh "pytest -n 2 --junitxml=pytest_result.xml"
                  junit "pytest_result.xml"
                }
              }
            }
          }
        }
        stage('HPF test') {
          steps {
            dir("${REPO}/test/test_hpf") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  sh "pytest --junitxml=pytest_result.xml"
                  junit "pytest_result.xml"
                }
              }
            }
          }
        }
        stage('Pipeline tests') {
          steps {
            dir("${REPO}/test/pipeline") {
              withEnv(["hydra_audio_PATH=/projects/hydra_audio"]) {
                withEnv(["PIPELINE_FULL_RUN=${PIPELINE_FULL_RUN}", "SENSORY_PATH=${env.WORKSPACE}/sensory_sdk/", "AMAZON_WWE_PATH=${env.WORKSPACE}/amazon_wwe/"]) {
                  withTools(params.TOOLS_VERSION) {
                    withVenv {
                      echo "PIPELINE_FULL_RUN set as " + env.PIPELINE_FULL_RUN

                      // Note we have 2 xcore targets and we can run x86 threads too. But in case we have only xcore jobs in the config, limit to 4 so we don't timeout waiting for xtags
                      sh "pytest -n 4 --junitxml=pytest_result.xml -vv"
                      junit "pytest_result.xml"
                      sh "python compare_keywords.py results_Avona_aec_ic_prev_arch_xcore.csv results_Avona_aec_ic_prev_arch_python.csv --pass-threshold=1"
                    }
                  }
                }
              }
            }
          }
        }
        stage('Benchmark Pipeline test results') {
          when {
            expression { env.PIPELINE_FULL_RUN == "1" }
          }
          steps {
            dir("${REPO}/test/pipeline") {
              withTools(params.TOOLS_VERSION) {
                withVenv {
                  copyArtifacts filter: '**/results_*.csv', fingerprintArtifacts: true, projectName: '../lib_audio_pipelines/master', selector: lastSuccessful()
                  runPython("python plot_results.py lib_audio_pipelines/tests/pipelines/results_lib_ap_prev_arch_xcore.csv results_Avona_prev_arch_xcore.csv --single-plot --ww-column='0_2 1_2' --figname=results_benchmark_prev_arch")
                  runPython("python plot_results.py lib_audio_pipelines/tests/pipelines/results_lib_ap_alt_arch_xcore.csv results_Avona_alt_arch_xcore.csv --single-plot --ww-column='0_2 1_2' --figname=results_benchmark_alt_arch")                    
                }
              }
            }
          }
        }
      }// stages
      post {
        always {
          // Examples artifacts
          archiveArtifacts artifacts: "${REPO}/build/**/fwk_voice_example_bare_metal_*", fingerprint: true
          // AEC aretfacts
          archiveArtifacts artifacts: "${REPO}/test/lib_adec/test_adec_profile/**/adec_prof*.log", fingerprint: true
          // IC artefacts
          archiveArtifacts artifacts: "${REPO}/test/lib_ic/test_ic_profile/ic_prof.log", fingerprint: true
          archiveArtifacts artifacts: "${REPO}/test/lib_ic/test_ic_spec/ic_spec_summary.txt", fingerprint: true
          // NS artefacts
          archiveArtifacts artifacts: "${REPO}/test/lib_ns/test_ns_profile/ns_prof.log", fingerprint: true
          // VNR artifacts
          archiveArtifacts artifacts: "${REPO}/test/lib_vnr/test_wav_vnr/*.png", fingerprint: true
          archiveArtifacts artifacts: "${REPO}/test/lib_vnr/test_wav_vnr/*.csv", fingerprint: true
          archiveArtifacts artifacts: "${REPO}/examples/bare-metal/vnr/*.png", fingerprint: true
          archiveArtifacts artifacts: "${REPO}/examples/bare-metal/vnr/vnr_prof.log", fingerprint: true
          // Pipelines tests
          archiveArtifacts artifacts: "${REPO}/test/pipeline/**/results_*.csv", fingerprint: true
          archiveArtifacts artifacts: "${REPO}/test/pipeline/**/results_*.png", fingerprint: true, allowEmptyArchive: true
          archiveArtifacts artifacts: "${REPO}/test/pipeline/keyword_input_*/*.npy", fingerprint: true, allowEmptyArchive: true
        }
        failure {
          // archive wavs on failure only
          archiveArtifacts artifacts: "${REPO}/test/pipeline/keyword_input_*/*.wav", fingerprint: true
        }
        cleanup {
          xcoreCleanSandbox()
        }
      }
    }// stage xcore.ai Verification
  }
}

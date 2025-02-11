XMOS Voice Framework Change Log
===============================

0.8.1

  * FIXED: Added back missing documentation

0.8.0
-----

  * CHANGED: Tools version from 15.2.1 to 15.3.0
  * CHANGED: Updated xmos-ai-tools version from 0.1.8 to 1.3.1
  * CHANGED: Updated lib_xcore_math version from 2.1.1 to 2.4.0

0.7.0
-----

  * CHANGED: Tools version from 15.1.4 to 15.2.1
  * CHANGED: Example builds and docs use Ninja instead of nmake under Windows
  * CHANGED: Update xmos_xmake_toolchain to v1.0.0 from untagged commit
             3a19f0284c66a92dbb9d5adc9d3d5016aac22646

0.6.0
-----

  * CHANGED: Improved documentation style
  * CHANGED: Replace lib_xs3_math with the lib_xcore_math v2.1.1
  * CHANGED: Integrate new version of lib_tflite_micro in VNR module

0.5.1
-----

  * ADDED: Windows documentation
  * REMOVED: VAD module
  * CHANGED: Git hash at which lib_tflite_micro is fetched during CMake FetchContent

0.5.0
-----

  * ADDED: Support for VNR
  * CHANGED: VNR input based IC control system (the API is not backwards compatible)
  * CHANGED: VNR input based AGC in pipeline examples
  * ADDED: Amazon based wake word engine testing in piplines tests

0.4.0
-----

  * CHANGED: Increased ASR AGC amplitude target
  * ADDED: -Os compile option for modules, examples and tests

0.3.0
-----

  * ADDED: Support for VAD.
  * CHANGED: xcore_sdk no longer a submodule of avona.

0.2.0
-----

  * ADDED: Support for IC, NS and ADEC.
  * CHANGED: CMake files cleanup.

0.1.0
-----

  * Initial version with support for AEC and AGC libraries.


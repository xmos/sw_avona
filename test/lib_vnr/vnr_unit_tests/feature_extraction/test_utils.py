
import numpy as np
import xscope_fileio
import xtagctl
import os
import tempfile
import sys
import scipy.io.wavfile
import math
import subprocess
from pathlib import Path

from xmos_ai_tools.xinterpreters import TFLMHostInterpreter

sys.path.append(str(Path(__file__).parents[3] / "shared" / "python"))
import py_vs_c_utils as pvc

def get_model():
    return str(Path(__file__).parents[4] / "modules" / "lib_vnr" / "python" / "model" / "model_output" / "trained_model.tflite")

def get_vnr_conf():
    return Path(__file__).parents[5] / "py_voice" / "py_voice" / "config" / "components" / "vnr_only.json"

def run_dut(input_data, test_name, xe):
    tmp_folder = tempfile.mkdtemp(dir=".", suffix=os.path.basename(test_name))
    prev_path = os.getcwd()
    os.chdir(tmp_folder)
    input_data.astype(np.int32).tofile("input.bin")

    if(os.path.splitext(xe)[-1] == ".xe"): # xcore run
        with xtagctl.acquire("XCORE-AI-EXPLORER") as adapter_id:
            xscope_fileio.run_on_target(adapter_id, xe)
    else: # x86 run
        subprocess.run([xe])
        
    with open("output.bin", "rb") as fdut:
        dut_output = np.fromfile(fdut, dtype=np.int32)
    os.chdir(prev_path)
    os.system("rm -r {}".format(tmp_folder))
    return dut_output

def double_to_int32(x, exp):
    y = x.astype(np.float64) * (2.0 ** -exp)
    y = y.astype(np.int32)
    return y

def int32_to_double(x, exp):
    y = x.astype(np.float64) * (2.0 ** exp)
    return y

def f64_to_float_s32(d):
    m,e = math.frexp(d)
    m_int = int(m * (2.0 ** 31))
    e = e-31
    return (m_int, e)

def get_closeness_metric(ref, dut):
    tmp_folder = tempfile.mkdtemp(dir=".")
    prev_path = os.getcwd()
    os.chdir(tmp_folder)
    output_file = "temp.wav"
    output_wav_data = np.zeros((2, len(ref)))
    output_wav_data[0,:] = ref
    output_wav_data[1,:] = dut
    scipy.io.wavfile.write(output_file, 16000, output_wav_data.T)
    arith_closeness, geo_closeness, c_delay, peak2ave = pvc.pcm_closeness_metric(output_file, verbose=False)
    os.chdir(prev_path)
    os.system("rm -r {}".format(tmp_folder))
    return arith_closeness, geo_closeness

def get_model_details(model_file):
    with TFLMHostInterpreter() as interpreter_tflite: # important to close interpreter to avoid OOM error
        interpreter_tflite.set_model(model_path=model_file)
        input_details = interpreter_tflite.get_input_details()[0]
        output_details = interpreter_tflite.get_output_details()[0]
    assert(input_details["dtype"] in [np.int8, np.uint8]), "Error: Need 8bit model for quantisation"
    assert(output_details["dtype"] in [np.int8, np.uint8]), "Error: Need 8bit model for quantisation"
    return input_details, output_details

def quantise_patch(this_patch, input_details):
    input_scale, input_zero_point = input_details["quantization"]
    this_patch = this_patch / input_scale + input_zero_point
    this_patch = np.round(this_patch)
    this_patch = np.clip(this_patch, np.iinfo(input_details["dtype"]).min, np.iinfo(input_details["dtype"]).max)
    this_patch = this_patch.astype(input_details["dtype"])
    return this_patch

def dequantise_output(output_data, output_details):
    output_scale, output_zero_point = output_details["quantization"]
    output_data = output_data.astype(np.float64)
    output_data = (output_data - output_zero_point) * output_scale
    return output_data

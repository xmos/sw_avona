# Copyright 2022 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from builtins import str
import pytest
import subprocess
import xtagctl


def pytest_collect_file(parent, path):
    if(path.ext == ".xe"):
        print('path = ',path)
        return UnityTestSource.from_parent(parent, fspath=path)

class UnityTestSource(pytest.File):
    def collect(self):
        # Find the binary built from the runner for this test file
        #
        # Assume the following directory layout:
        # unit_tests/       <- Test root directory
        # |-- bin/          <- Compiled binaries of the test runners
        # |-- conftest.py   <- This file
        # |-- runners/      <- Auto-generated buildable source of test binaries
        # |-- src/          <- Unity test functions
        # `-- wscript       <- Build system file used to generate/build runners
        print("self.name ", self.fspath)
        yield UnityTestExecutable.from_parent(self, fspath=self.fspath, name=self.name)


class UnityTestExecutable(pytest.Item):
    def __init__(self, fspath, name, parent):
        super(UnityTestExecutable, self).__init__(name, parent)
        self.fspath = fspath
        self._nodeid = self.name  # Override the naming to suit C better

    def runtest(self):
        # Run the binary in the simulator
        simulator_fail = False
        test_output = None
        try:
            print("run xrun for executable ", self.fspath)
            with xtagctl.acquire("XCORE-AI-EXPLORER") as adapter_id:
                test_output = subprocess.check_output(['xrun', '--xscope', '--io', '--adapter-id', adapter_id, self.fspath], text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            # Unity exits non-zero if an assertion fails
            simulator_fail = True
            test_output = e.output

        # Parse the Unity output
        unity_pass = False
        test_output = test_output.split('\n')
        for line in test_output:
            if 'test' in line:
                test_report = line.split(':')
                # Unity output is as follows:
                #   <test_source>:<line_number>:<test_case>:PASS
                #   <test_source>:<line_number>:<test_case>:FAIL:<failure_reason>
                test_source = test_report[0]
                line_number = test_report[1]
                test_case = test_report[2]
                result = test_report[3]
                failure_reason = None
                print(('\n {}()'.format(test_case)), end=' ')
                if result == 'PASS':
                    unity_pass = True
                    continue
                if result == 'FAIL':
                    failure_reason = test_report[4]
                    print('')  # Insert line break after test_case print
                    raise UnityTestException(self, {'test_source': test_source,
                                                    'line_number': line_number,
                                                    'test_case': test_case,
                                                    'failure_reason':
                                                        failure_reason})

        if simulator_fail:
            raise Exception(self, "Simulation failed.")
        #if not unity_pass:
        #    raise Exception(self, "Unity test output not found.")
        print('')  # Insert line break after final test_case which passed

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, UnityTestException):
            return '\n'.join([str(self.parent).strip('<>'),
                              '{}:{}:{}()'.format(
                                    excinfo.value[1]['test_source'],
                                    excinfo.value[1]['line_number'],
                                    excinfo.value[1]['test_case']),
                              'Failure reason:',
                              excinfo.value[1]['failure_reason']])
        else:
            return str(excinfo.value)

    def reportinfo(self):
        # It's not possible to give sensible line number info for an executable
        # so we return it as 0.
        #
        # The source line number will instead be recovered from the Unity print
        # statements.
        return self.fspath, 0, self.name


class UnityTestException(Exception):
    pass

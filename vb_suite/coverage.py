from suite import *
import os
import shutil

PATH = os.path.dirname(__file__)

TEST_DIR = os.path.join(PATH, 'vb_coverage')

if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

shutil.copyfile(os.path.join(PATH, 'pandas_vb_common.py'),
                os.path.join(TEST_DIR, 'pandas_vb_common.py'))

for bench in benchmarks:
    bench.to_testcase(TEST_DIR)

import os
import sys

from shutil import rmtree

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(TEST_DIR, os.pardir)
TMP_DIR = os.path.join(TEST_DIR, 'tmp')

def pytest_configure():
    sys.path.append(ROOT_DIR)

def pytest_sessionstart():
    os.mkdir(TMP_DIR)

def pytest_sessionfinish():
    rmtree(TMP_DIR)

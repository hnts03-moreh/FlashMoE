"""
Build script for rocshmem-python.

Usage:
    pip install .                          # standard install
    pip install -e .                       # editable install
    CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/rocm" pip install .  # custom ROCm path
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.fspath(Path(self.get_ext_fullpath(ext.name)).parent.resolve())
        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}/rocshmem",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]

        # Forward CMAKE_ARGS from environment
        env_cmake_args = os.environ.get("CMAKE_ARGS", "")
        if env_cmake_args:
            cmake_args.extend(env_cmake_args.split())

        build_dir = Path(self.build_temp) / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_dir, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--parallel"],
            cwd=build_dir, check=True
        )


setup(
    name="rocshmem-python",
    version="0.1.0",
    description="Python bindings for ROCSHMEM (AMD ROCm SHMEM)",
    packages=["rocshmem"],
    ext_modules=[CMakeExtension("rocshmem.core")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
)

import sys, re, os

try:
    from skbuild import setup
    import nanobind
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

setup(
    name="simple_cv_process_pywrapper",
    version="0.0.1",
    packages=["simple_cv_process_pywrapper"],
    package_dir={"": "src"},
    cmake_args=["-DCMAKE_BUILD_TYPE=Debug"],
    cmake_install_dir="src/simple_cv_process_pywrapper",
    include_package_data=True,
    python_requires=">=3.8"
)
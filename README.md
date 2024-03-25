# nanobind_opencv_exercise
![peppers](https://github.com/yuki-inaho/nanobind_opencv_exercise/blob/main/doc/output.png)

# Build
```
git submodule update --init --recursive
cd simple_cv_process_pywrapper
mkdir -p build && cd build
cmake .. && make -j
```

# Run an example
```
cd example
pip install -r requirements.txt
python example.py
```
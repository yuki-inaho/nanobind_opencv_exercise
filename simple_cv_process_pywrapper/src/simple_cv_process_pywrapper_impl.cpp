#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstring> // for std::memcpy

namespace nb = nanobind;

cv::Mat NDarrayToCVMat(nb::ndarray<uint8_t, nb::shape<nb::any, nb::any, 3>, nb::c_contig, nb::device::cpu> arr)
{
    // Assume that arr is C-contiguous
    return cv::Mat(arr.shape(0), arr.shape(1), CV_8UC3, arr.data());
}

nb::ndarray<nb::numpy, uint8_t, nb::shape<nb::any, nb::any, 3>> CVMatToNDarray(const cv::Mat &mat)
{
    uint8_t *data = new uint8_t[mat.total() * mat.elemSize()];
    std::memcpy(data, mat.data, mat.total() * mat.elemSize());
    size_t shape[3] = {mat.rows, mat.cols, mat.channels()};
    nb::capsule deleter(data, [](void *data) noexcept
                        { delete[] (uint8_t *)data; });
    return nb::ndarray<nb::numpy, uint8_t, nb::shape<nb::any, nb::any, 3>>(data, 3, shape, deleter);
}

NB_MODULE(simple_cv_process_pywrapper_impl, m)
{
    m.def(
        "bgr2rgb", [](const nb::ndarray<uint8_t, nb::shape<nb::any, nb::any, 3>, nb::c_contig, nb::device::cpu> &image_bgr)
        {
            cv::Mat image_bgr_mat = NDarrayToCVMat(image_bgr).clone();
            cv::Mat image_rgb_mat;
            cv::cvtColor(image_bgr_mat, image_rgb_mat, cv::COLOR_BGR2RGB);
            return CVMatToNDarray(image_rgb_mat); });
}

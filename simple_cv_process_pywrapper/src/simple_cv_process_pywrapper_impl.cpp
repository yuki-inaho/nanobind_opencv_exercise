#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstring> // for std::memcpy
#include <random>

namespace nb = nanobind;

template <typename T, std::size_t Channels>
cv::Mat NDarrayToCVMat(nb::ndarray<T, nb::shape<nb::any, nb::any, Channels>, nb::c_contig, nb::device::cpu> arr)
{
    int cvType;
    if constexpr (Channels == 3)
    {
        cvType = std::is_same<T, uint8_t>::value ? CV_8UC3 : CV_16SC3;
    }
    else if constexpr (Channels == 1)
    {
        cvType = std::is_same<T, uint8_t>::value ? CV_8UC1 : CV_16SC1;
    }
    else
    {
        static_assert(Channels == 1 || Channels == 3, "Only 1 or 3 channels are supported.");
    }
    return cv::Mat(arr.shape(0), arr.shape(1), cvType, arr.data());
}

template <typename T, std::size_t Channels>
nb::ndarray<nb::numpy, T, nb::shape<nb::any, nb::any, Channels>> CVMatToNDarray(const cv::Mat &mat)
{
    T *data = new T[mat.total() * mat.elemSize()];
    std::memcpy(data, mat.data, mat.total() * mat.elemSize());

    size_t shape[3];
    if constexpr (Channels == 3)
    {
        shape[0] = mat.rows;
        shape[1] = mat.cols;
        shape[2] = mat.channels();
    }
    else if constexpr (Channels == 1)
    {
        shape[0] = mat.rows;
        shape[1] = mat.cols;
        // Do not set shape[2] as it's beyond the bounds for 1-channel images.
    }
    else
    {
        static_assert(Channels == 1 || Channels == 3, "Only 1 or 3 channels are supported.");
    }

    nb::capsule deleter(data, [](void *data) noexcept
                        { delete[] static_cast<T *>(data); });
    return nb::ndarray<nb::numpy, T, nb::shape<nb::any, nb::any, Channels>>(data, Channels == 3 ? 3 : 2, shape, deleter);
}

nb::ndarray<nb::numpy, int16_t, nb::shape<nb::any, nb::any, 1>> generateRandomNoiseImageInt16(const nb::ndarray<uint8_t, nb::shape<nb::any, nb::any, 3>, nb::c_contig, nb::device::cpu> &inputImage)
{
    // Determine the size from the input image
    size_t rows = inputImage.shape(0);
    size_t cols = inputImage.shape(1);

    // Create a new cv::Mat for the noise image
    cv::Mat noiseImage(rows, cols, CV_16SC1);

    // Fill the noise image with normally distributed random values
    cv::randn(noiseImage, cv::Scalar::all(0), cv::Scalar::all(1000));

    // Convert the cv::Mat back to an ndarray to be returned
    return CVMatToNDarray<int16_t, 1>(noiseImage);
}

NB_MODULE(simple_cv_process_pywrapper_impl, m)
{
    m.def("bgr2rgb", [](const nb::ndarray<uint8_t, nb::shape<nb::any, nb::any, 3>, nb::c_contig, nb::device::cpu> &image_bgr)
          {
        cv::Mat image_bgr_mat = NDarrayToCVMat<uint8_t, 3>(image_bgr).clone();
        cv::Mat image_rgb_mat;
        cv::cvtColor(image_bgr_mat, image_rgb_mat, cv::COLOR_BGR2RGB);
        return CVMatToNDarray<uint8_t, 3>(image_rgb_mat); });

    m.def("bgr2gray", [](const nb::ndarray<uint8_t, nb::shape<nb::any, nb::any, 3>, nb::c_contig, nb::device::cpu> &image_bgr)
          {
        cv::Mat image_bgr_mat = NDarrayToCVMat<uint8_t, 3>(image_bgr).clone();
        cv::Mat image_gray_mat;
        cv::cvtColor(image_bgr_mat, image_gray_mat, cv::COLOR_BGR2GRAY);
        return CVMatToNDarray<uint8_t, 1>(image_gray_mat); });
    m.def("generate_int16_noise_image", &generateRandomNoiseImageInt16, nb::arg("input_image").noconvert());
}

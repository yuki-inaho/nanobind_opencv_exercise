#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <iostream>

namespace nb = nanobind;
using namespace nb::literals;

using NBTensor8UC3 = nb::tensor<nb::numpy, uint8_t, nb::shape<nb::any, nb::any, 3>, nb::c_contig, nb::device::cpu>;
using NBTensor8UC1 = nb::tensor<nb::numpy, uint8_t, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;

/* @TODO: add type and shape validation
   @TODO: use std::copy?
*/

cv::Mat ConvertNBTensorToCVMat8CU3(NBTensor8UC3 &tensor)
{
    int32_t image_width = tensor.shape(0);
    int32_t image_height = tensor.shape(1);
    cv::Mat cv_mat = cv::Mat::zeros(cv::Size(image_width, image_height), CV_8UC3);
    for (size_t v = 0; v < image_height; ++v)
    {
        for (size_t u = 0; u < image_width; ++u)
        {
            for (size_t ch = 0; ch < 3; ++ch)
            {
                cv_mat.at<cv::Vec3b>(v, u)[ch] = (uint8_t)tensor(v, u, ch);
            }
        }
    }
    return cv_mat;
}

NBTensor8UC3 ConvertCVMat8CU3ToNBTensor(const cv::Mat &cv_mat)
{
    int32_t image_width = cv_mat.cols;
    int32_t image_height = cv_mat.rows;
    size_t shape[3] = {image_width, image_height, 3};

    uint8_t *data = new uint8_t[image_width * image_height * 3]{0};
    nb::capsule deleter(data, [](void *data) noexcept
                        { delete[](uint8_t *) data; });
    NBTensor8UC3 tensor(data, 3, shape, deleter);
    for (size_t v = 0; v < image_height; ++v)
    {
        for (size_t u = 0; u < image_width; ++u)
        {
            for (size_t ch = 0; ch < 3; ++ch)
            {
                tensor(v, u, ch) = cv_mat.at<cv::Vec3b>(v, u)[ch];
            }
        }
    }
    return tensor;
}

NBTensor8UC3 ConvertBGR2RGB(NBTensor8UC3 &image_ndarray)
{
    int32_t image_width = image_ndarray.shape(1);
    int32_t image_height = image_ndarray.shape(0);

    cv::Mat image_cv_rgb;
    cv::Mat image_cv_bgr = ConvertNBTensorToCVMat8CU3(image_ndarray);
    cv::cvtColor(image_cv_bgr, image_cv_rgb, cv::COLOR_BGR2RGB);
    return ConvertCVMat8CU3ToNBTensor(image_cv_rgb);
}

NB_MODULE(simple_cv_process_pywrapper_impl, m)
{
    m.def(
        "bgr2rgb", &ConvertBGR2RGB,
        "image_ndarray"_a);
}

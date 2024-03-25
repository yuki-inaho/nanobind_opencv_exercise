import cv2
import numpy as np
from pathlib import Path
import simple_cv_process_pywrapper as cvp

SCRIPT_DIR = Path(__file__).parent.resolve()


def main():
    image_path = SCRIPT_DIR / "data" / "peppers.png"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # Ensure the image is read correctly
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert the OpenCV BGR image to RGB using the C++ function
    image_rgb = cvp.bgr2rgb(image_bgr)

    # Convert the OpenCV BGR image to Grayscale using the C++ function
    image_gray = cvp.bgr2gray(image_bgr)

    # Display the original BGR, converted RGB, and Grayscale images side by side
    concatenated_image = cv2.hconcat([image_bgr, image_rgb, cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)])
    cv2.imshow("BGR vs RGB vs Gray", concatenated_image)

    # Generate int16 noise image based on the size of the input image
    noise_image = cvp.generate_int16_noise_image(image_bgr)

    # Convert int16 noise to a displayable format by normalizing and converting to uint8
    displayable_noise = cv2.cvtColor(
        cv2.normalize(noise_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLOR_GRAY2BGR,
    )
    concatenated_noise = cv2.hconcat([image_bgr, displayable_noise])
    cv2.imshow("RGB vs Noise", concatenated_noise)
    cv2.imwrite("output.png", concatenated_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

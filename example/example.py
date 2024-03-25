import cv2
from pathlib import Path
import simple_cv_process_pywrapper as cvp

SCRIPT_DIR = Path(__file__).parent.resolve()


def main():
    image_path = SCRIPT_DIR / "data" / "peppers.png"
    image_bgr = cv2.imread(str(image_path))

    # Ensure the image is read correctly
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert the OpenCV BGR image to RGB using the C++ function.
    image_rgb = cvp.bgr2rgb(image_bgr)

    # Display the original BGR image and the converted RGB image side by side
    concatenated_image = cv2.hconcat([image_bgr, image_rgb])
    cv2.imshow("BGR vs RGB", concatenated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

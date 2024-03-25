import cv2
import numpy as np
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
    # Note: OpenCV images are NumPy arrays, so this conversion is straightforward.
    image_rgb = cvp.bgr2rgb(image_bgr)

    # Convert back to uint8 if necessary. This step might be redundant depending
    # on how your C++ function is implemented, but is a common step when manipulating images.
    
    import pdb; pdb.set_trace()
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    # Display the original BGR image and the converted RGB image side by side
    concatenated_image = cv2.hconcat([image_bgr, image_rgb])
    cv2.imshow("BGR vs RGB", concatenated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from pathlib import Path
import simple_cv_process_pywrapper

SCRIPT_DIR = str(Path(__file__).parent.resolve())


def main():
    image_path = str(Path(SCRIPT_DIR, "data", "peppers.png"))
    image_bgr = cv2.imread(image_path)
    image_rgb = simple_cv_process_pywrapper.bgr2rgb(image_bgr)
    cv2.imshow("exercise", cv2.hconcat([image_bgr, image_rgb]))
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
from mss import mss
import numpy as np
import cv2
from fastai.vision.all import *
import time


# Screen capture
with mss() as screen:
    screen.monitor = screen.monitors[0]
    while True:
        vision = np.array(screen.grab(screen.monitor))



        cv2.imshow("Meme Detector", vision)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q')]:
            break


cv2.destroyAllWindows()


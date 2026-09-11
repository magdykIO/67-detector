from mss import mss
import numpy as np
import cv2
from fastai.vision.all import *
import time


learn = load_learner(Path(__file__).with_name("67.pkl"))

# Screen capture
with mss() as screen:
    screen.monitor = screen.monitors[0]
    while True:
        vision = np.array(screen.grab(screen.monitor))
        rgb = cv2.cvtColor(vision, cv2.COLOR_BGRA2RGB)
        pred, pred_idx, probs = learn.predict(PILImage.create(rgb))
        cv2.putText(
            vision,
            f"{pred} {float(probs[pred_idx]):.2f}",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3,
        )

        cv2.imshow("Meme Detector", vision)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q')]:
            break


cv2.destroyAllWindows()


import pathlib
import sys

# 67.pkl was exported on Windows / a newer Python pathlib layout.
sys.modules.setdefault("pathlib._local", pathlib)
pathlib.WindowsPath = pathlib.PosixPath

from mss import mss
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from fastai.vision.all import *


learn = load_learner(Path(__file__).with_name("67.pkl"), cpu=True)
learn.model.eval()
labels = list(learn.dls.vocab)
mean = torch.tensor(imagenet_stats[0])[:, None, None]
std = torch.tensor(imagenet_stats[1])[:, None, None]


def classify(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    x = ((x - mean) / std).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(learn.model(x), dim=1)[0]
    idx = int(probs.argmax())
    return labels[idx], float(probs[idx])


# Screen capture
with mss() as screen:
    screen.monitor = screen.monitors[0]
    while True:
        vision = np.array(screen.grab(screen.monitor))
        pred, conf = classify(vision)
        cv2.putText(
            vision,
            f"{pred} {conf:.2f}",
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

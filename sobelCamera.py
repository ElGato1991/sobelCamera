import cv2 as cv
import numpy as np
import pyvirtualcam

cap = cv.VideoCapture(1)  # ggf. Index anpassen
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS) or 30

with pyvirtualcam.Camera(width=w, height=h, fps=int(fps), print_fps=True) as cam:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Sobel (x & y), Magnitude -> Schwarz-Weiß
        sx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
        sy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        mag = cv.magnitude(sx, sy)

        # Normieren auf 0..255 und zu RGB expandieren
        mag = np.uint8(255 * (mag / (mag.max() + 1e-6)))
        sobel_rgb = cv.cvtColor(mag, cv.COLOR_GRAY2RGB)

        cam.send(sobel_rgb)
        cam.sleep_until_next_frame()

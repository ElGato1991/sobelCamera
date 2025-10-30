import cv2 as cv
import numpy as np
import pyvirtualcam
import msvcrt  # Für nicht-blockierende Tastatureingaben (Windows)

cap = cv.VideoCapture(1)  # ggf. Index anpassen
if not cap.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden. Bitte Geräteindex prüfen.")

w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS) or 30

print("Modus wechseln: [1] Magnitude, [2] Sobel X, [3] Sobel Y | [q] Beenden")
mode = 1  # 1= Magnitude, 2= X, 3= Y

with pyvirtualcam.Camera(width=w, height=h, fps=int(fps), print_fps=True) as cam:
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Sobel (x & y)
            sx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
            sy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

            # Ausgabe je nach Modus wählen
            if mode == 1:
                # Magnitude aus x & y
                mag = cv.magnitude(sx, sy)
                out = np.uint8(255 * (mag / (mag.max() + 1e-6)))
            elif mode == 2:
                # Nur Sobel X
                ax = np.abs(sx)
                out = np.uint8(255 * (ax / (ax.max() + 1e-6)))
            else:
                # Nur Sobel Y
                ay = np.abs(sy)
                out = np.uint8(255 * (ay / (ay.max() + 1e-6)))

            sobel_rgb = cv.cvtColor(out, cv.COLOR_GRAY2RGB)

            cam.send(sobel_rgb)
            cam.sleep_until_next_frame()

            # Tastatureingabe ohne Fenster abfragen (nur Windows)
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'1', b'2', b'3'):
                    mode = int(ch.decode())
                    if mode == 1:
                        print("Modus: Magnitude")
                    elif mode == 2:
                        print("Modus: Sobel X")
                    elif mode == 3:
                        print("Modus: Sobel Y")
                elif ch in (b'q', b'Q'):
                    print("Beende…")
                    break
    finally:
        cap.release()

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

print("Modus: [1] Magnitude, [2] Sobel X, [3] Sobel Y | Farben: [4] Grün, [5] Rot, [6] Gelb, [0] Normal | [Leertaste/N] Nächste Farbe | [C] Farbe an/aus | [M] Matrix (grün) | [R] Ziffern auf dunklen Flächen | [ [ / ] ] Farben schieben | [Q] Beenden")
mode = 1  # 1= Magnitude, 2= X, 3= Y

# Farbdarstellung (Regenbogen) für die Ausgabe ein-/ausschalten
# 'c' Taste zum Umschalten. Standard: EIN (gewünscht: Regenbogenfarben)
colorize = True

# Farbverschiebung (0..255) zum zyklischen Verschieben der Colormap
color_shift = 0


# Matrix-Darstellung (grün auf schwarz) umschalten
matrix_mode = False

# Weitere Einzelkanal-Farben
red_mode = False
yellow_mode = False

# Zyklische Farbauswahl (Colormap -> Grün -> Rot -> Gelb -> Graustufen)
modes = ["colormap", "green", "red", "yellow", "grayscale"]
current_mode_idx = 0  # start mit Colormap

# Bevorzugten Colormap wählen (TURBO, falls verfügbar; sonst JET)
try:
    DEFAULT_COLORMAP = cv.COLORMAP_TURBO
except AttributeError:
    DEFAULT_COLORMAP = cv.COLORMAP_JET
current_colormap = DEFAULT_COLORMAP

# Ziffern-Overlay auf dunklen Bereichen (aus instanceDetection.py übernommen)
digits_on_dark = False
digits_font = cv.FONT_HERSHEY_SIMPLEX
digits_scale = 0.45
digits_thickness = 1
digits_color = (0, 255, 0)  # grün
_tsize, _tbase = cv.getTextSize('8', digits_font, digits_scale, digits_thickness)
digit_w, digit_h = _tsize[0], _tsize[1]
cell_w = max(8, digit_w + 4)
cell_h = max(10, digit_h + 6)
darkness_thresh = 30  # mittlere Helligkeit < Schwelle => "schwarze" Fläche

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

            # Einzelkanal-Farben (Grün/Rot/Gelb), Colormap oder Graustufen ausgeben
            if matrix_mode or red_mode or yellow_mode:
                zeros = np.zeros_like(out)
                if matrix_mode:
                    # Grün auf schwarz
                    sobel_rgb = np.dstack((zeros, out, zeros))  # RGB
                elif red_mode:
                    # Rot auf schwarz
                    sobel_rgb = np.dstack((out, zeros, zeros))  # RGB
                else:
                    # Gelb (Rot + Grün) auf schwarz
                    sobel_rgb = np.dstack((out, out, zeros))  # RGB
            elif colorize:
                # applyColorMap erwartet 8-bit Single-Channel
                # Durch Addition (uint8 wrap-around) die Farben zyklisch verschieben
                out_shifted = np.uint8(out + color_shift)
                colored = cv.applyColorMap(out_shifted, current_colormap)  # BGR
                sobel_rgb = cv.cvtColor(colored, cv.COLOR_BGR2RGB)
            else:
                sobel_rgb = cv.cvtColor(out, cv.COLOR_GRAY2RGB)

            # Optional: kleine grüne Ziffern auf dunklen Bereichen des aktuellen Bildes zeichnen
            if digits_on_dark:
                gray_view = cv.cvtColor(sobel_rgb, cv.COLOR_RGB2GRAY)
                for yy in range(0, h, cell_h + 2):
                    for xx in range(0, w, cell_w + 2):
                        roi = gray_view[yy:yy + cell_h, xx:xx + cell_w]
                        if roi.size == 0:
                            continue
                        if roi.mean() < darkness_thresh:
                            ch = str(np.random.randint(0, 10))
                            org = (xx, yy + digit_h)
                            cv.putText(sobel_rgb, ch, org, digits_font, digits_scale, digits_color, digits_thickness, cv.LINE_AA)

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
                elif ch in (b'c', b'C'):
                    colorize = not colorize
                    # Wenn Colormap aktiviert wird, auf Modus "colormap" springen, sonst Graustufen
                    if colorize:
                        matrix_mode = red_mode = yellow_mode = False
                        current_mode_idx = 0
                        print("Modus: Colormap")
                    else:
                        matrix_mode = red_mode = yellow_mode = False
                        current_mode_idx = 4
                        print("Modus: Graustufen")
                elif ch in (b'm', b'M'):
                    matrix_mode = not matrix_mode
                    if matrix_mode:
                        red_mode = False
                        yellow_mode = False
                        colorize = False
                        current_mode_idx = 1
                    print(f"Matrix-Stil: {'AN (grün/schwarz)' if matrix_mode else 'AUS'}")
                elif ch in (b'r', b'R'):
                    # Ziffern-Overlay auf dunklen Bereichen umschalten
                    digits_on_dark = not digits_on_dark
                    print(f"Ziffern auf dunklen Flächen: {'AN' if digits_on_dark else 'AUS'}")
                elif ch in (b'4',):
                    matrix_mode, red_mode, yellow_mode = True, False, False
                    colorize = False
                    current_mode_idx = 1
                    print("Farbwahl: Grün (Matrix)")
                elif ch in (b'5',):
                    matrix_mode, red_mode, yellow_mode = False, True, False
                    colorize = False
                    current_mode_idx = 2
                    print("Farbwahl: Rot")
                elif ch in (b'6',):
                    matrix_mode, red_mode, yellow_mode = False, False, True
                    colorize = False
                    current_mode_idx = 3
                    print("Farbwahl: Gelb")
                elif ch in (b'0',):
                    matrix_mode = red_mode = yellow_mode = False
                    colorize = True
                    current_mode_idx = 0
                    print("Farbwahl: Normal (Colormap)")
                elif ch in (b'[', b'{'):
                    color_shift = (color_shift - 8) % 256
                    print(f"Farbverschiebung: {color_shift}")
                elif ch in (b']', b'}'):
                    color_shift = (color_shift + 8) % 256
                    print(f"Farbverschiebung: {color_shift}")
                elif ch in (b' ', b'n', b'N'):
                    # Nächste Farbe durchschalten
                    current_mode_idx = (current_mode_idx + 1) % len(modes)
                    mode_name = modes[current_mode_idx]
                    matrix_mode = red_mode = yellow_mode = False
                    if mode_name == "colormap":
                        colorize = True
                        print("Modus: Colormap")
                    elif mode_name == "green":
                        colorize = False
                        matrix_mode = True
                        print("Farbwahl: Grün (Matrix)")
                    elif mode_name == "red":
                        colorize = False
                        red_mode = True
                        print("Farbwahl: Rot")
                    elif mode_name == "yellow":
                        colorize = False
                        yellow_mode = True
                        print("Farbwahl: Gelb")
                    else:  # grayscale
                        colorize = False
                        print("Modus: Graustufen")
                elif ch in (b'q', b'Q'):
                    print("Beende…")
                    break
    finally:
        cap.release()
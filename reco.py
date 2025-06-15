import cv2
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ultralytics import YOLO

# Load pose model (includes hands as part of pose points)
model = YOLO('yolov8n-pose.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volbar = 400
volper = 0
volMin, volMax = volume.GetVolumeRange()[:2]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for pose in results.keypoints:
        keypoints = pose.data[0].cpu().numpy()

        # Use index 4 (thumb tip) and 8 (index fingertip) for hand gesture (similar to MediaPipe)
        try:
            x1, y1 = int(keypoints[4][0]), int(keypoints[4][1])
            x2, y2 = int(keypoints[8][0]), int(keypoints[8][1])

            cv2.circle(frame, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            length = hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [30, 350], [volMin, volMax])
            volbar = np.interp(length, [30, 350], [400, 150])
            volper = np.interp(length, [30, 350], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            # Volume bar and percentage display
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 255), 4)
            cv2.rectangle(frame, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
        except IndexError:
            pass

    cv2.imshow("Hand Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()

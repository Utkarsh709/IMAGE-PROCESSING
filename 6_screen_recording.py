# Screen Recorder

import cv2 as c
import pyautogui as p
import numpy as np

# create resolution
rs = p.size()   # (width, height)

# filename in which we store recording
fn = input("please enter any file name and path: ")

# fix the frame rate
fps = 60.0
fourcc = c.VideoWriter_fourcc(*"XVID")
output = c.VideoWriter(fn, fourcc, fps, rs)

# create recording window
c.namedWindow("Live Recording", c.WINDOW_NORMAL)
c.resizeWindow("Live Recording", (600, 400))

while True:
    img = p.screenshot()
    f = np.array(img)

    # convert RGB → BGR for OpenCV
    f = c.cvtColor(f, c.COLOR_RGB2BGR)

    output.write(f)
    c.imshow("Live Recording", f)

    if c.waitKey(1) & 0xFF == ord('q'):
        break

output.release()
c.destroyAllWindows()

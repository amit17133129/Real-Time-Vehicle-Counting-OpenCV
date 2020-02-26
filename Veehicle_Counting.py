import numpy as np
import cv2
import imutils


class Camera:

    @staticmethod
    def start_capturing(stream_path):
        cap = cv2.VideoCapture(stream_path)

        while (cap.isOpened()):
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


import argparse
import datetime
import imutils
import math
import cv2
import numpy as np


width = 1000

textIn = 0
textOut = 0

def testIntersectionIn(x, y):
    res = -450 * x + 450* y + 157500
    if ((res >= -550) and (res < 550)):
        print(str(res))
        return True
    return False


def testIntersectionOut(x, y):
    res = -450 * x + 400 * y + 180000
    if ((res >= -550) and (res <= 550)):
        print(str(res))
        return True

    return False


if __name__ == "__main__":
    camera = cv2.VideoCapture("----file_path----")

    firstFrame = None

    # loop over the frames of the video
    while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        (grabbed, frame) = camera.read()
        text = "Unoccupied"


        if not grabbed:
            break


        frame = imutils.resize(frame, width=width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)


        if firstFrame is None:
            firstFrame = gray
            continue


        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=5)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for c in cnts:
            print(c)

            if cv2.contourArea(c) < 120:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.line(frame, (width // 2, 200), (width, 500), (250, 0, 1), 2)  # blue line
            cv2.line(frame, (width // 2 -250, 0), (width - 50, 500), (0, 0, 255), 2)  # red line

            rectagleCenterPont = ((x + x + w) // 2, (y + y + h) // 2)
            cv2.circle(frame, rectagleCenterPont, 1, (0, 0, 255), 5)

            if (testIntersectionIn((x + x + w) // 2, (y + y + h) // 2)):
                textIn += 1

            if (testIntersectionOut((x + x + w) // 2, (y + y + h) // 2)):
                textOut += 1


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.putText(frame, "In: {}".format(str(textIn)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Out: {}".format(str(textOut)), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.imshow("Vehicle Counter", frame)

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

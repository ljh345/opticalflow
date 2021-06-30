#!/usr/bin/env python3

import numpy as np
import cv2 as cv

def opticalflow(x, outdir):
    cap = cv.VideoCapture(x)
    ret, image = cap.read()

    rgb_weights = [0.2989, 0.5870, 0.1140]
    prev_frame = np.dot(image[...,:3], rgb_weights)

    mask = np.zeros_like(image)
    mask[..., 1] = 255

    count = 0
    while(cap.isOpened()):

        # Read sequence of frames and convert to grayscale.
        ret, frame = cap.read()
        frame = np.dot(frame[...,:3], rgb_weights)

        # Calculate Dense Optical Flow between previous and current frames using Farneback method.
        flow = cv.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])           # Computes magnitude and angle of 2D vectors.

        mask[..., 0] = angle * 180 / np.pi / 2                                  # Sets image value according to optical flow direction.
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)    # Sets image value to the optical flow magnitude (normalized)
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)                               # Converts HSV to RGB (BGR) representation

        cv.imwrite(outdir + '/output_oflow_{:04d}.png'.format(count), rgb)
        print("Processed frame:", count)

        prev_frame = frame
        count += 1

    cap.release()
    cv.destroyAllWindows()

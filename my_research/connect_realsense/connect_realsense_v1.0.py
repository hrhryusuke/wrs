import cv2
import pickle
import struct
import numpy as np
import drivers.devices.realsense_rpi.realsense_client as realsense_client

rc = realsense_client.EtherSenseClient(address="10.2.0.202", port=18360)
cv2.namedWindow("window")
while True:
    cv2.imshow("window", rc.get_rgb_image())
    key = cv2.waitKey(1)
    if key != -1:
        break

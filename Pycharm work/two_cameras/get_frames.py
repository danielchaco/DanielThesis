import argparse
import time
from datetime import datetime
import os
from time import monotonic
from uuid import uuid4
from multiprocessing import Process, Queue
import cv2
import depthai as dai
import contextlib

fps = 10

now = datetime.now()
today = now.strftime("%m/%d/%Y")
today_time = now.strftime("%m/%d/%Y %H_%M_%S")

current_dir = r'C:\Users\Daniel\PycharmProjects\Daniel_Thesis\two_cameras'
main_folder = os.path.join(current_dir,today)
test_folder = os.path.join(main_folder,today_time)

def getPipeline(device_type):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    rgb = pipeline.create(dai.node.ColorCamera)
    rgb.setPreviewSize(600, 300)
    rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP) #THE_1080_P
    rgb.setInterleaved(False)
    rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    rgb.setFps(fps)

    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName('control')
    controlIn.out.link(rgb.inputControl)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    rgbOut.setStreamName("rgb")
    rgb.preview.link(rgbOut.input)


    # Monocameras
    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setFps(fps)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setFps(fps)

    # depth = pipeline.create(dai.node.StereoDepth)
    # depth.initialConfig.setConfidenceThreshold(255)
    # median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    # depth.initialConfig.setMedianFilter(median)
    # depth.setLeftRightCheck(False)
    # depth.setExtendedDisparity(False)
    # depth.setSubpixel(False)

    # left.out.link(depth.left)
    # right.out.link(depth.right)

    # Create output
    leftOut = pipeline.create(dai.node.XLinkOut)
    leftOut.setStreamName("left")
    left.out.link(leftOut.input)
    rightOut = pipeline.create(dai.node.XLinkOut)
    rightOut.setStreamName("right")
    right.out.link(rightOut.input)
    # depthOut = pipeline.create(dai.node.XLinkOut)
    # depthOut.setStreamName("disparity")
    # depth.disparity.link(depthOut.input)

    return pipeline

q_rgb_list = []
q_left_list = []
q_right_list = []
# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()
    usbSpeed = dai.UsbSpeed.SUPER
    openvino_version = dai.OpenVINO.Version.VERSION_2021_4
    device_name =[]

    if len(device_infos) == 0:
        raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")

    for device_info in device_infos:
        # Note: the pipeline isn't set here, as we don't know yet what device it is.
        # The extra arguments passed are required by the existing overload variants

        device = stack.enter_context(dai.Device(openvino_version, device_info, usbSpeed))

        # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
        # print("=== Connected to " + device_info.getMxId())
        # device_name.append(device_info.getMxId())
        mxid = device.getMxId()
        # cameras = device.getConnectedCameras()
        # usb_speed = device.getUsbSpeed()
        # print("   >>> MXID:", mxid)
        # print("   >>> Cameras:", *[c.name for c in cameras])
        # print("   >>> USB speed:", usb_speed.name)

        device_type = "OAK-D"
        # if   len(cameras) == 1: device_type = "OAK-1"
        # elif len(cameras) == 3: device_type = "OAK-D"
        # # If USB speed is UNKNOWN, assume it's a POE device
        # if usb_speed == dai.UsbSpeed.UNKNOWN: device_type += "-POE"

        # Get a customized pipeline based on identified device type
        pipeline = getPipeline(device_type)
        # print("   >>> Loading pipeline for:", device_type)
        device.startPipeline(pipeline)

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        stream_name = "rgb-" + mxid + "-" + device_type
        q_rgb_list.append((q_rgb, stream_name))

        # Output queue will be used to get the left and right frames
        q_left = device.getOutputQueue(name = 'left')
        stream_name = "left-" + mxid + "-" + device_type
        q_left_list.append((q_left,stream_name))
        q_right = device.getOutputQueue(name='right')
        stream_name = "right-" + mxid + "-" + device_type
        q_right_list.append((q_right,stream_name))

    while True:
        for q_rgb, stream_name in q_rgb_list:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                cv2.imshow(stream_name, in_rgb.getCvFrame())
        for q_left, stream_name in q_left_list:
            in_left = q_left.tryGet()
            if in_left is not None:
                cv2.imshow(stream_name, in_left.getCvFrame())
        for q_right, stream_name in q_right_list:
            in_right = q_right.tryGet()
            if in_right is not None:
                cv2.imshow(stream_name, in_right.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
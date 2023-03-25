#!/usr/bin/env python3
# videos recording? pitch, yaw and roll
import cv2
import numpy as np
import depthai as dai
import time
import pandas as pd
import os

color = (255, 255, 255)
desired_aruco_dictionary = 'DICT_6X6_50' #'DICT_6X6_250' #'DICT_ARUCO_ORIGINAL' #"DICT_6X6_250"
size_of_marker = 0.0145 # side lenght of the marker in meter
length_of_axis = 0.01
width, height = 1280, 720
time_data, id_data, coors_data = [],[],[]
today = time.strftime("%d %b %Y %H_%M_%S ",time.localtime()).strip()
folder = os.path.join(r'C:\Users\Daniel\PycharmProjects\Daniel_Thesis\Data','Test_'+today)
img_folder = os.path.join(folder,'video')
os.makedirs(img_folder,exist_ok=True)

# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
fps = 20

# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P

# Create pipeline
pipeline = dai.Pipeline()
# pipeline.setXLinkChunkSize(0)

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

rgbOut = pipeline.create(dai.node.XLinkOut)
disparityOut = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

rgbOut.setStreamName("rgb")
disparityOut.setStreamName("disp")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

#Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(rgbResolution)
camRgb.setFps(fps)
if downscaleColor: camRgb.setIspScale(2, 3)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
camRgb.initialControl.setManualFocus(130)

left.setResolution(monoResolution)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(fps)
right.setResolution(monoResolution)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.setExtendedDisparity(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Initial Config ROI
topLeft = dai.Point2f(.49, .49)
bottomRight = dai.Point2f(.51, .51)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(disparityOut.input)

spatialLocationCalculator.passthroughDepth.link(disparityOut.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# The different ArUco dictionaries built into the OpenCV library.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    found = True
    frameRgb = None
    # frameDisp = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbWindowName = "rgb"
    # depthWindowName = "depth"
    # blendedWindowName = "rgb-depth"
    cv2.namedWindow(rgbWindowName)
    # cv2.namedWindow(depthWindowName)
    # cv2.namedWindow(blendedWindowName)
    # cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)

    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    # Load ArUco Dict and params
    this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()
    calibData = device.readCalibration()
    M_rgb, _, _ = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
    # print(width,height)
    mtx = np.array(M_rgb)
    D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
    dist = np.array(D_rgb)
    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        queueEvents = device.getQueueEvents(("rgb", "disp")) #
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            t = time.time()

            frameRgb = latestPacket["rgb"].getCvFrame()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frameRgb, this_aruco_dictionary,
                                                               parameters=this_aruco_parameters)
            frame_markers = cv2.aruco.drawDetectedMarkers(frameRgb, corners, ids)
            rvecs, tvecs, trash = cv2.aruco.estimatePoseSingleMarkers(corners, size_of_marker, mtx, dist)
            imaxis = cv2.aruco.drawDetectedMarkers(frameRgb, corners, ids)
            if tvecs is not None:
                roi_list = []
                for i in range(len(tvecs)):
                    imaxis = cv2.aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)
                    rvec = np.squeeze(rvecs[0], axis=None)
                    tvec = np.squeeze(tvecs[0], axis=None)
                    tvec = np.expand_dims(tvec, axis=1)
                    rvec_matrix = cv2.Rodrigues(rvec)[0]
                    proj_matrix = np.hstack((rvec_matrix, tvec))
                    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                    corner = corners[i]
                    # x_mid = (corner[0][1][0] + corner[0][3][0]) / 2
                    # y_mid = (corner[0][1][1] + corner[0][3][1]) / 2
                    corner = corner.reshape((4, 2))
                    # (top_left, top_right, bottom_right, bottom_left) = corner
                    x_mid = np.mean(corner[:, 0])
                    y_mid = np.mean(corner[:, 1])
                    # if found:
                    #     print(x_mid,y_mid)
                    topLeft = dai.Point2f((x_mid - 10) / width, (y_mid - 10) / height)
                    bottomRight = dai.Point2f((x_mid + 10) / width, (y_mid + 10) / height)
                    config = dai.SpatialLocationCalculatorConfigData()
                    config.depthThresholds.lowerThreshold = 100
                    config.depthThresholds.upperThreshold = 10000
                    config.roi = dai.Rect(topLeft, bottomRight)
                    roi_list.append(config)
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.setROIs(roi_list)
                spatialCalcConfigInQueue.send(cfg)
                spatialData = spatialCalcQueue.get().getSpatialLocations()
                coors = []
                for depthData in spatialData:
                    roi = depthData.config.roi
                    roi = roi.denormalize(width=frameRgb.shape[1], height=frameRgb.shape[0])
                    xmin = int(roi.topLeft().x)
                    ymin = int(roi.topLeft().y)
                    xmax = int(roi.bottomRight().x)
                    ymax = int(roi.bottomRight().y)
                    # fontType = cv2.FONT_HERSHEY_TRIPLEX
                    # # cv2.rectangle(frameRgb, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                    # cv2.putText(frameRgb, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 10),
                    #             fontType, 0.3, (0, 255, 255))
                    # cv2.putText(frameRgb, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 20),
                    #             fontType, 0.3, (0, 255, 255))
                    # cv2.putText(frameRgb, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 30),
                    #             fontType, 0.3, (0, 255, 255))
                    coors.append([int(depthData.spatialCoordinates.x),int(depthData.spatialCoordinates.y),int(depthData.spatialCoordinates.z)])
                # if found:
                #     print(ids,len(ids),type(ids),ids[0],ids[0][0],ids[3][0])
                #     print(coors)
                if ids is not None and len(ids) == len(coors):
                    time_data.append(t)
                    id_data.append(list(ids))
                    coors_data.append(coors)
                    cv2.imwrite(os.path.join(img_folder,f'{t*1000}.jpg'), frameRgb)
                if found:
                    # print(coors)
                    # print(coors_data)
                    found = False
            # Show the frame
            cv2.imshow(rgbWindowName, frameRgb)
        if cv2.waitKey(1) == ord('q'):
            # task.stop()
            # task.close()
            break
if len(coors_data) > 1:
    if len(coors_data) == len(id_data) == len(time_data):
        os.makedirs(folder,exist_ok=True)
        df = pd.DataFrame()
        df['time'] = time_data
        # IDs = np.unique(np.concatenate(id_data))
        for i in range(len(id_data)):
            for j,ID in enumerate(id_data[i]):
                # print(ID[0],coors_data[i][j])
                for k,COR in enumerate(['_X','_Y','_Z']):
                    # print(coors_data[i])
                    df.at[i,str(ID[0])+COR] = coors_data[i][j][k]
        # print(df)
        path = os.path.join(folder, 'ArUco_Data.csv')
        # df.to_csv(folder+'ArUco_Data.csv',index=False)
        df.to_csv(path,index=False)
    else:
        print('ERROR: Data info does not match')
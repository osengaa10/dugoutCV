#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
# OAK-D
# 14442C1061E6C1D200 XLinkDeviceState.X_LINK_UNBOOTED

# OAK-1
# 14442C10E12853D000 XLinkDeviceState.X_LINK_UNBOOTED

for device in dai.Device.getAllAvailableDevices():
    print(f"{device.getMxId()} {device.state}")

device_info = dai.Device.getDeviceByMxId("14442C1061E6C1D200")
device_info2 = dai.Device.getDeviceByMxId("14442C10E12853D000")



'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
'''

# Tiny yolo v3/4 label texts


labelMap = ["ball"]

syncNN = True

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/frozen_darknet_yolov4_model.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_1)

# Define a source - color camera
colorCam = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutRgb = pipeline.createXLinkOut()
xoutNN = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")


colorCam.setPreviewSize(416, 416)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(255)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)
# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(1)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
spatialDetectionNetwork.setIouThreshold(0.5)

# Create outputs

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    colorCam.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

x_coordinates_set = []
y_coordinates_set = []
z_coordinates_set = []

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline, device_info) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []

    startTime = time.monotonic()
    counter = 0
    fps = 0
    # color = (255, 255, 255)
    color = (0, 0, 0)
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []

    t_coordinates = []


    while True:
        inPreview = previewQueue.get()
        inNN = detectionNNQueue.get()
        depth = depthQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        detections = inNN.detections
        # print(len(detections))

        if len(detections) != 0:
            boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
            roiDatas = boundingBoxMapping.getConfigData()
            ## FIGURING OUT THE DETECTION QUEUE STUFF
            # if len(detections) > 1:
            #     for detection in detections:
            #         print("x: " + str(detection.spatialCoordinates.x))
            #         print("y: " + str(detection.spatialCoordinates.y))
            #         print("z: " + str(detection.spatialCoordinates.z))
            #         print(int(time.time() * 1000))
            #     print("=========================")
            #     print("=========================")

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width = frame.shape[1]

        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            # # For straight on trajectory
            # x_coordinates.append(detection.spatialCoordinates.z/1000)
            # y_coordinates.append(detection.spatialCoordinates.y/1000)
            # z_coordinates.append(detection.spatialCoordinates.x/1000)
            # t_coordinates.append(int(time.time() * 1000)/1000)

            # For side trajectory
            x_coordinates.append(detection.spatialCoordinates.x / 1000)
            y_coordinates.append(detection.spatialCoordinates.y / 1000)
            z_coordinates.append(detection.spatialCoordinates.z / 1000)
            t_coordinates.append(int(time.time() * 1000)/1000)

            x_coordinates_set.append(detection.spatialCoordinates.x / 1000)
            y_coordinates_set.append(detection.spatialCoordinates.y / 1000)
            z_coordinates_set.append(detection.spatialCoordinates.z / 1000)
            print("x: " + str(x_coordinates_set))
            print("y: " + str(y_coordinates_set))
            print("z: " + str(z_coordinates_set))
            if len(x_coordinates) > 2:
                x_coordinates.pop(0)
                y_coordinates.pop(0)
                z_coordinates.pop(0)
                t_coordinates.pop(0)
                #x_coordinates_set.append(x_coordinates)
            # if len(x_coordinates) == 2:
            #     trajectories(x_coordinates, y_coordinates, z_coordinates, t_coordinates)

            # if len(x_coordinates) == 2:
            #     x_coordinates_set.append(x_coordinates)
            #     x_coordinates =[]
            # print(x_coordinates_set)
            # print("x: " + str(detection.spatialCoordinates.x))
            # print("y: " + str(detection.spatialCoordinates.y))
            # print("z: " + str(detection.spatialCoordinates.z))
            # print(int(time.time() * 1000))
            # print(detection.label)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break

# print("Done!")
# # Creating figure
# fig = plt.figure(figsize=(10, 7))
# ax = plt.axes(projection="3d")
#
# # Creating plot
# ax.scatter3D(x_coordinates_set, y_coordinates_set, z_coordinates_set, color="green")
# ax.plot(x_coordinates_set,y_coordinates_set,z_coordinates_set, color='r')
# plt.title("simple 3D scatter plot")
# ax.set_xlabel('X-axis', fontweight ='bold')
# ax.set_ylabel('Y-axis', fontweight ='bold')
# ax.set_zlabel('Z-axis', fontweight ='bold')
# plt.show()
#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
from statistics import mean
import time
import matplotlib.pyplot as plt
import socket
# sys.path.insert(0, "./kinematics")
# import trajectory
from kinematics.trajectory import trajectories
from kinematics.trajectory import vectors
'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
'''

# Tiny yolo v3/4 label texts

def sendVectors(vx, vy, vz):
    s = socket.socket()
    # connect to the server on local computer
    s.connect(('192.168.1.123', 1755))
    s.send((
        str(vz) + "," +
        str(vy) + "," +
        str(-vx)).encode())
    s.close()

def vectors(x,y,z,t):
    # theta = math.degrees(math.atan((z[1] - z[0])/(y[1] - y[0])))
    # print("theta: " + str(theta))
    v0x = ((x[1] - x[0])/(t[1] - t[0]))
    v0y = ((y[1] - y[0])/(t[1] - t[0]))
    v0z = ((z[1] - z[0])/(t[1] - t[0]))
    print("vx: " + str(v0x) + " vy: " + str(v0y) + " vz: " + str(v0z))
    return v0x, v0y, v0z

labelMap = ["ball"]

syncNN = True


# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/frozen_darknet_yolov4_model2.blob')).resolve().absolute())
# nnBlobPath = str((Path(__file__).parent / Path('models/frozen_darknet_yolov4_model_416x416_13shaves.blob')).resolve().absolute())
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
colorCam.initialControl.setManualFocus(129)
spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
objectTracker = pipeline.createObjectTracker()

xoutRgb = pipeline.createXLinkOut()
trackerOut = pipeline.createXLinkOut()

# xoutNN = pipeline.createXLinkOut()
# xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
# xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
trackerOut.setStreamName("tracklets")
# xoutNN.setStreamName("detections")
# xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
# xoutDepth.setStreamName("depth")

controlIn = pipeline.createXLinkIn()
controlIn.setStreamName('control')
controlIn.out.link(colorCam.inputControl)

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
# spatialDetectionNetwork.setNumNCEPerInferenceThread(19)
# spatialDetectionNetwork.setNumInferenceThreads(4)
# print(spatialDetectionNetwork.getNumInferenceThreads())

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(1)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
spatialDetectionNetwork.setIouThreshold(0.5)

# spatialDetectionNetwork.setNumInferenceThreads(2)
# print(spatialDetectionNetwork.getNumInferenceThreads())

# Create outputs

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)

# TINKERING
# objectTracker.setMaxObjectsToTrack(1)

objectTracker.setDetectionLabelsToTrack([0])  # track only ball
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

objectTracker.out.link(trackerOut.input)

spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)
# if syncNN:
#     spatialDetectionNetwork.passthrough.link(xoutRgb.input)
# else:
#     colorCam.preview.link(xoutRgb.input)

# spatialDetectionNetwork.out.link(xoutNN.input)
# spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
# spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

x_coordinates_set = []
y_coordinates_set = []
z_coordinates_set = []
t_coordinates_set = []
i = 0
vx_list = []
vy_list = []
vz_list = []
# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Start pipeline
    controlQueue = device.getInputQueue('control')
    device.startPipeline()

    previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    tracklets = device.getOutputQueue(name="tracklets", maxSize=1, blocking=False)

    frame = None

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    # color = (0, 0, 0)
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []

    t_coordinates = []


    while True:
        inPreview = previewQueue.get()
        track = tracklets.get()

        ctrl = dai.CameraControl()
        ctrl.setManualExposure(1000, 1600)
        controlQueue.send(ctrl)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()
        trackletsData = track.tracklets
        detected_time = time.monotonic()

        if len(trackletsData) != 0:
            t_coordinates.append(detected_time)
            x_coordinates.append(trackletsData[0].spatialCoordinates.x / 1000)
            y_coordinates.append(trackletsData[0].spatialCoordinates.y / 1000)
            z_coordinates.append(trackletsData[0].spatialCoordinates.z / 1000)
            # print(trackletsData[0].status)
            # print(trackletsData[0].id)
            print(x_coordinates)

            if len(x_coordinates) > 2:
                x_coordinates.pop(0)
                y_coordinates.pop(0)
                z_coordinates.pop(0)
                t_coordinates.pop(0)

            if len(x_coordinates) == 2:
                v0x = ((x_coordinates[1] - x_coordinates[0]) / (t_coordinates[1] - t_coordinates[0]))
                v0y = ((y_coordinates[1] - y_coordinates[0]) / (t_coordinates[1] - t_coordinates[0]))
                v0z = ((z_coordinates[1] - z_coordinates[0]) / (t_coordinates[1] - t_coordinates[0]))
                print("x: " + str(x_coordinates))
                print("x_delta: " + str(x_coordinates[1] - x_coordinates[0]))
                print("t: " + str(t_coordinates))
                print("t_delta: " + str(t_coordinates[1] - t_coordinates[0]))
                # print("vx: " + str(v0x) + " vy: " + str(v0y) + " vz: " + str(v0z))
                print("vx: " + str(v0x))
                vx_list.append(v0x)
                vy_list.append(v0y)
                vz_list.append(v0z)


        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)


            print("vx_list: " + str(vx_list))
            try:
                label = labelMap[t.label]
            except:
                label = t.label
            statusMap = {dai.Tracklet.TrackingStatus.NEW: "NEW", dai.Tracklet.TrackingStatus.TRACKED: "TRACKED",
                         dai.Tracklet.TrackingStatus.LOST: "LOST"}
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, statusMap[t.status], (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color)
            cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color)
            cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                    color)

        cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord('q'):
            # sendVectors(mean(vx_list), mean(vy_list), mean(vz_list))
            print("mean: " + str(mean(vx_list)))
            print("x_coordinates_set length: " + str(len(x_coordinates_set)))
            print(x_coordinates_set)
            print("x_coordinates length: " + str(len(x_coordinates)))
            print(x_coordinates)
            break
        # if cv2.waitKey(1) == ord('q') or len(vx_list) > 2:
        #     # sendVectors(mean(vx_list), mean(vy_list), mean(vz_list))
        #     print("mean: " + str(mean(vx_list)))
        #     break










print("Done!")
# # Creating figure
# fig = plt.figure(figsize=(10, 7))
# ax = plt.axes(projection="3d")
# ax.set_xlim([-2,2])
# ax.set_ylim([-2,2])
# ax.set_zlim([-2,2])
# # Creating plot
# ax.scatter3D(x_coordinates_set, y_coordinates_set, z_coordinates_set, color="green")
# ax.plot(x_coordinates_set,y_coordinates_set,z_coordinates_set, color='r')
# plt.title("simple 3D scatter plot")
# ax.set_xlabel('X-axis', fontweight ='bold')
# ax.set_ylabel('Y-axis', fontweight ='bold')
# ax.set_zlabel('Z-axis', fontweight ='bold')
# plt.show()
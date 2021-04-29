# import math
from pathlib import Path
import numpy as np
import cv2
import depthai as dai

coco_labels = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

p = dai.Pipeline()

camRgb = p.createColorCamera()
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Send bouding box from the NN to host via XLink
rgb_xout = p.createXLinkOut()
rgb_xout.setStreamName("rgb")
camRgb.preview.link(rgb_xout.input)

# NN that detects faces in the image
face_nn = p.createNeuralNetwork()
face_nn.setBlobPath(str(Path("models/yolov5s_opt_20213_6shaves.blob").resolve().absolute()))
camRgb.preview.link(face_nn.input)

# Send bouding box from the NN to host via XLink
nn_xout = p.createXLinkOut()
nn_xout.setStreamName("nn")
face_nn.out.link(nn_xout.input)

def nms_python(detections, nms_thresh):
    # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
    # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
    # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
    # boxes = [r.box for r in regions]
    boxes = []
    scores = []
    for det in detections:
        if det[5] > 0.4:
            boxes.append(det[:4])
            scores.append(det[5])

               #         x, y, w, h = det[:4]
            #         x_min = int(416*(x-w/2))
            #         y_min = int(416*(y-h/2))
            #         x_max = int(416*(x+w/2))
            #         y_max = int(416*(y+h/2))
            # 0,1,2,3 ->box,4->confidence，5-85 -> coco classes confidence
    boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]
    # [[-67, 284, 289, 289], [-29, 295, 258, 258], [-28, 292, 270, 270], [-71, 302, 299, 299], [-38, 297, 273, 273], [-35, 295, 279, 279], [-60, 277, 317, 317]]
    scores = [r.pd_score for r in regions]
    # [0.572265625, 0.8251953125, 0.69384765625, 0.5263671875, 0.82666015625, 0.669921875, 0.5615234375]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
    return [regions[i[0]] for i in indices]

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    device.startPipeline()
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    while True:
        frame = qRgb.get().getCvFrame()
        inNn = qNn.tryGet()

        if inNn is not None:
            inference = np.array(inNn.getFirstLayerFp16()).reshape(10647,85)
            print(nms_python(inference, 0.2))
            # nms_python
            # for i in range(10):
            #     det = dets[i]
            #     #0,1,2,3 ->box,4->confidence，5-85 -> coco classes confidence
            #     conf = det[4]
            #     if conf > 0.3:
            #         x, y, w, h = det[:4]
            #         x_min = int(416*(x-w/2))
            #         y_min = int(416*(y-h/2))
            #         x_max = int(416*(x+w/2))
            #         y_max = int(416*(y+h/2))
            #         coco = det[5:]
            #         label = coco_labels[np.argmax(coco)]
            #         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #         cv2.putText(frame, label, (x_min + 5, y_min + 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            #         print(label)
            #         print(det)
                # print(identities)

        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
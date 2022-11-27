# python3 rtod.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from functions import escort, shoot

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True)
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-c", "--confidence", type=float, default=0.2)
args = vars(ap.parse_args())

turn, shootPermission = True, False
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Инициализация...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("Подготовка к работе...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	net.setInput(blob)
	detections = net.forward()

	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			if label[:-7] == "person:":
				print('Обнаружен человек!')
				escort()
				if shootPermission:
					shoot()
	if turn:
		print('Готов к работе!')
		turn = False

	cv2.imshow("RTOD", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	fps.update()
fps.stop()

cv2.destroyAllWindows()
vs.stop()
#import libraries
import os   
import cv2
import numpy as np

def get_boxes(image_path, model):
  # split file name and extension
  filename, extension = os.path.splitext(image_path)
  if (extension in ['.png', '.jpg', '.jpeg']):
    # read image using cv2
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    
    # input to layer
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass through model
    model.setInput(blob)
    detections = model.forward()

    boxes = []
    confidences = []

    count = 0
    for i in range(0, detections.shape[2]):
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")

      confidence = detections[0, 0, i, 2]

      if (confidence > 0.165):
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        count = count + 1
        boxes.append(box)
        confidences.append(confidence)
    
  cv2.imwrite('temp_output.jpg', image)
  print("Face detection complete for image " + image_path + " (" + str(count) + ") faces found!")
  return (boxes, confidences)

    
# model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
# (boxes, confidences) = get_boxes('images/dad-in-san-fran.jpg', model)
# print(boxes)
# print(confidences)
# box = boxes[0]
# minx, miny, maxx, maxy = box
# image = cv2.imread('temp_output.jpg')
# cropped = image[int(miny):int(maxy), int(minx):int(maxx)]
# cv2.imshow('name', cropped)
# cv2.imwrite('just_face.jpg', cropped)
# cv2.waitKey(0)
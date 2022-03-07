import numpy as np
import time
import cv2


class_list = ['person']
img_size = 416


def engine_backend():
    engine = cv2.dnn.readNetFromONNX('./best-416.onnx')
    engine.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    engine.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    return engine


def preprocess(image):
    row, col, _ = image.shape
    _max = max(col,row)
    resized = np.zeros((_max,_max,3), np.uint8)
    resized[0:row, 0:col] = image

    x = cv2.dnn.blobFromImage(resized, 1/255.0, (img_size,img_size), swapRB = True)
    return x, resized



def postprocess(image, predictions, score_thresh = 0.45, nms_thresh = 0.45):
    class_ids = []
    confidences = []
    boxes = []

    rows = predictions.shape[0]

    image_width, image_height, _ = image.shape

    x_factor = image_width / img_size
    y_factor =  image_height / img_size

    for r in range(rows):
        row = predictions[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()

                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, nms_thresh) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_confidences, result_class_ids, result_boxes


def draw_box(image, result_confidences, result_class_ids, result_boxes):
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]

        color = colors[class_id % len(colors)]

        conf  = result_confidences[i]
        label = "{} {}".format(class_list[class_id], round(float(conf),2))

        cv2.rectangle(image, box, color, 1)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(image, label, (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
    return image




engine = engine_backend()
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0


if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    x, resized = preprocess(frame)
    engine.setInput(x)
    predictions = engine.forward()

    result_confidences, result_class_ids, result_boxes = postprocess(resized, predictions[0])
    frame = draw_box(frame, result_confidences, result_class_ids, result_boxes)

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    fps = str(int(fps))

    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('output', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
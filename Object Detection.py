import numpy as np
import argparse
import cv2
import os
import time
import math

from itertools import chain

SAFE_DISTANCE = 170

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []
    lines = []
    box_centers = []

    for output in outputs:
        for detection in output:
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                box_centers = [centerX, centerY]

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):

    survcount = 0
    viccount = 0

    if len(idxs) > 0:
        unsafe =[]
        count = 0

        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            centeriX = boxes[i][0] + (boxes[i][2] // 2)
            centeriY = boxes[i][1] + (boxes[i][3] // 2)

            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])

            idxs_copy = list(idxs.flatten())
            idxs_copy.remove(i)

            for j in np.array(idxs_copy):
                centerjX = boxes[j][0] + (boxes[j][2] // 2)
                centerjY = boxes[j][1] + (boxes[j][3] // 2)

                distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centeriY, 2))
                #cv2.putText(image, str(distance), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if distance <= SAFE_DISTANCE:
                    cv2.line(image, (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1]  + (boxes[i][3] // 2)), (boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2)), (0, 0, 255), 2)
                    unsafe.append([centerjX, centerjY])
                    unsafe.append([centeriX, centeriY])

            if centeriX in chain(*unsafe) and centeriY in chain(*unsafe):
                count += 1

            if str(labels[classIDs[i]]) == 'Survivor':
                survcount += 1
            if str(labels[classIDs[i]]) == 'Victim':
                viccount += 1

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(image, (0, 450), (235, 385), (0, 0, 0), -1)
            cv2.putText(image, 'Social distancing Violators: {}'.format(count), (0, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 127), 2)
            cv2.putText(image, 'Survivors: {}' .format(survcount), (0, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            cv2.putText(image, 'Victims: {}'.format(viccount), (0, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image


def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]

    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default='custom-yolov4-detector_best4300.weights', help='Path to model weights')
    parser.add_argument('-cfg', '--config', type=str, default='custom-yolov4-detector.cfg', help='Path to configuration file')
    parser.add_argument('-l', '--labels', type=str, default='coco.names', help='Path to label file')
    parser.add_argument('-c', '--confidence', type=float, default=0.9, help='Minimum confidence for a box to be detected.')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for Non-Max Suppression')
    parser.add_argument('-u', '--use_gpu', default=True, action='store_true', help='Use GPU (OpenCV must be compiled for GPU). For more info checkout: https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/')
    parser.add_argument('-s', '--save', default=True, action='store_true', help='Whether or not the output should be saved')
    parser.add_argument('-sh', '--show', default=True, action="store_false", help='Show output')

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--image_path', type=str, default='', help='Path to the image file.')
    input_group.add_argument('-v', '--video_path', type=str, default='', help='Path to the video file.')

    args = parser.parse_args()

    # Get the labels
    labels = open(args.labels).read().strip().split('\n')

    # Create a list of colors for the labels
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load weights using OpenCV
    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)

    if args.use_gpu:
        print('Using GPU')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if args.save:
        print('Creating output directory if it doesn\'t already exist')
        os.makedirs('output', exist_ok=True)

    # Get the ouput layer names
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    if args.image_path != '':
        image = cv2.imread(args.image_path)

        boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, args.confidence, args.threshold)

        image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)

        # show the output image
        if args.show:
            cv2.imshow('YOLO Object Detection', image)
            cv2.waitKey(0)

        if args.save:
            cv2.imwrite(f'output/{args.image_path.split("/")[-1]}', image)
    else:
        if args.video_path != '':
            cap = cv2.VideoCapture(args.video_path)
        else:
            cap = cv2.VideoCapture('http://192.168.43.42/stream.jpg')

        if args.save:
            width = 600 #int(cap.get(3))
            height = 450 #int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            name = args.video_path.split("/")[-1] if args.video_path else 'camera.avi'
            out = cv2.VideoWriter(f'output/{name}', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

        while cap.isOpened():
            ret, image = cap.read()
            image=np.rot90(image)
            image=np.rot90(image)
            image=cv2.resize(image,(600,450))

            if not ret:
                print('Video file finished.')
                break

            boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, args.confidence, args.threshold)

            image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)

            if args.show:
                cv2.imshow('YOLO Object Detection', image)
                for i in range(3):
                    cap.grab()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.save:
                out.write(image)

        cap.release()
        if args.save:
            out.release()
    cv2.destroyAllWindows()

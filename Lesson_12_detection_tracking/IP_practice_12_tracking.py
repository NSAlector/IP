#!/usr/bin/env python
import numpy as np
import cv2

from functools import reduce

class Target:
    def __init__(self):
        #self.capture = cv.CaptureFromCAM(0)
        self.path_to_save = '003.csv'
        self.capture = cv2.VideoCapture('003.mp4')
        self.CLASSES =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
           "diningtable",  "dog", "horse", "motorbike", "SIM",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.LEGACY_CLASSES = [6,7]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
        self.RESIZED_DIMENSIONS = (300, 300)  # Dimensions that SSD was trained on.
        self.IMG_NORM_RATIO = 0.007843
        #self.bbox_colors = np.random.uniform(255, 0, size=(len(self.CLASSES), 3))
        self.bounding_box_list = []
        self.classes_list = []
        self.tracker_list = []
        self.labels_list = []
        self.object_lifetime = []

        cv2.namedWindow("Target", 1)
    #def get_h_frame(self,image):
    #    image = cv2.GaussianBlur(image, (15, 15), 0)
    #    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #    image, _, _ = cv2.split(image)
    #    return image
    def check_continue(self, size):
        for i in range(len(self.bounding_box_list)):
            boxA = self.bounding_box_list[i]
            xA = boxA[0]
            xB = boxA[0] + boxA[2]
            yA = boxA[1]
            yB = boxA[1] + boxA[3]
            if (xA <= 0.05*size[1] or xB>= 0.95*size[1]): # and
            #if    (yA <= 0.2*size[0] or yB>= 0.8*size[0]):
                self.labels_list[i] = False
    def is_merged(self, boxA,boxB):
        cx_A = boxA[0] + boxA[2] / 2
        cy_A = boxA[1] + boxA[3] / 2
        cr_A = max(boxA[2], boxA[3]) / 2

        cx_B = boxB[0] + boxB[2] / 2
        cy_B = boxB[1] + boxB[3] / 2
        cr_B = max(boxB[2], boxB[3]) / 2

        dist = (cx_A - cx_B) ** 2 + (cy_A - cy_B) ** 2
        return dist < (cr_A**2 + cr_B**2) -50
    def merge_boxes(self,image):
        i = 0
        while(i < len(self.bounding_box_list)):
            j = i+1
            while(j < len(self.bounding_box_list)):
                #print(len(self.bounding_box_list),i,j)
                if(self.classes_list[i]==self.classes_list[j] and self.labels_list[i] and self.labels_list[j]):
                    box_A = self.bounding_box_list[i]
                    box_B = self.bounding_box_list[j]

                    if self.is_merged(box_A,box_B):
                        bbox = [min(box_A[0], box_B[0]),
                                    min(box_A[1], box_B[1]),
                                    max(box_A[2] + box_A[0], box_B[2] + box_B[0]),
                                    max(box_A[3] + box_A[1], box_B[3] + box_B[1])]

                        bbox[2] -= bbox[0]
                        bbox[2] = max(bbox[2],1)
                        bbox[3] -= bbox[1]
                        bbox[3] = max(bbox[3], 1)
                        self.bounding_box_list[i] = bbox
                        #print(bbox)
                        self.bounding_box_list.pop(j)
                        self.tracker_list.pop(j)
                        self.classes_list.pop(j)
                        self.labels_list.pop(j)
                        self.object_lifetime.pop(j)
                        if(bbox[3] > 1 and bbox[2] > 1):
                            self.tracker_list[i] = cv2.legacy.TrackerCSRT_create()
                            self.tracker_list[i].init(image, bbox)
                        else:
                            self.labels_list[i] = False
                            i = i+1
                            j = i+1
                    else:
                        j += 1
                else:
                    j += 1
            i += 1


    def run(self):
        _, frame = self.capture.read()
        frame = cv2.resize(frame, (300,300), interpolation=cv2.INTER_LINEAR)
        frame_size = frame.shape[:2]
        sum = 0
        count_prev = 0
        count_cur = 0
        ret = True
        while ret:
            ret,frame = self.capture.read()
            if ret:
                frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_LINEAR)
                self.check_continue(frame.shape[:2])
                h,w = frame.shape[:2]
                color_image = frame.copy()
                if (len(self.bounding_box_list) > 0):
                    for i in range(len(self.bounding_box_list)):
                        if(self.labels_list[i]):
                            success, self.bounding_box_list[i] = self.tracker_list[i].update(frame)
                            if(success):
                                self.object_lifetime[i] += 1
                                (startX, startY, endX, endY) = np.array(self.bounding_box_list[i]).astype(int)
                                endX += startX
                                endY += startY
                                cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,0), cv2.FILLED)
                                label = "{}: {:.2f}%".format(self.classes_list[i], confidence * 100)
                                cv2.rectangle(color_image, (startX, startY), (
                                    endX, endY), self.COLORS[self.classes_list[i]], 2)
                                cv2.putText(color_image, self.CLASSES[self.classes_list[i]], (startX, startY), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, self.COLORS[self.classes_list[i]], 5)
                #cv2.imshow("Preprocessed", frame)
                frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, self.RESIZED_DIMENSIONS),
                                                   self.IMG_NORM_RATIO, self.RESIZED_DIMENSIONS, 127.5)

                self.net.setInput(frame_blob)
                output = self.net.forward()
                #t_bboxes = []
                for i in np.arange(output.shape[2]):
                    confidence = output[0, 0, i, 2]
                    # Confidence must be at least 30%
                    if confidence > 0.5:
                        idx = int(output[0, 0, i, 1])
                        if idx in self.LEGACY_CLASSES:
                            bounding_box = output[0, 0, i, 3:7] * np.array(
                                [w, h, w, h])
                            bbox = bounding_box.astype("int")
                            bbox[2] -= bbox[0]
                            bbox[3] -= bbox[1]
                            #t_bboxes.append(bbox)

                            self.tracker_list.append(cv2.legacy.TrackerCSRT_create())

                            self.bounding_box_list.append(bbox)
                            self.tracker_list[-1].init(color_image,bbox)
                            self.classes_list.append(idx)
                            self.labels_list.append(True)
                            self.object_lifetime.append(0)
                self.merge_boxes(color_image)
                cv2.imshow("Target", color_image)
                #print(len(self.bounding_box_list))
                count_cur = len([self.labels_list[i] for i in range(len(self.labels_list)) if self.labels_list[i]==True and
                                 self.object_lifetime[i] > 1])
                if(count_cur > count_prev):
                    sum += count_cur - count_prev
                count_prev = count_cur
                print(count_cur,sum)
                # Listen for ESC key
                c = cv2.waitKey(7) % 0x100
                if c == 2:
                    break
        np.savetxt(self.path_to_save, (['SIM'], [sum]), delimiter=',', fmt='%s')
if __name__=="__main__":
    t = Target()
    t.run()
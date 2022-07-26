import os
import cv2
import time
import torch
import numpy as np
from sort import *

#TODO: Find motion vector to predict motion?

VIDEO_PATH = "./videos/highway_1.mp4"
MODEL_PATH = "./models/yolov5m.pt"
BG_PATH = "./bg.jpg"

ppm = 10
fps = 15
car_w = 16
car_l = 32
if __name__ == "__main__":
    tracker = Sort(30, 1, 0.1)
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    model = torch.hub.load('./yolov5/', 'custom', path=MODEL_PATH, source="local")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    individual_count = 0
    db = {0: None, }
    velocities = {0: None}

    graph = cv2.resize(cv2.imread(BG_PATH), (1280, 720))

    while(cap.isOpened()):
        t = time.time()
        ret, frame = cap.read()
        if ret != True:
            break
        h, w, c = frame.shape
        pts1 = np.array([[0, h//3], [w, h//3], [0, h], [w, h]], np.float32)
        pts2 = np.array([[0, 0], [w, 0], [int(w*0.42) - 40, h], [int(w*0.58) - 40, h]], np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        new_frame = np.copy(frame)
        new_graph = np.copy(graph)
        if frame_count > 13*30 + 4:
            individual_count += 1
            if individual_count > 3:
                individual_count = -2
                continue
            elif individual_count < 1:
                continue
            detections = []
            
            if 13*30 + 4 < frame_count:
                result = model(frame, size=320)
                
                for p, pic in enumerate(result.xyxy):
                    for result in pic:
                        x1, y1, x2, y2, conf, c = result
                        x1, y1, x2, y2, conf, c = int(x1), int(y1), int(x2), int(y2), float(conf), float(c)
                        if conf >= 0.6:
                            detections.append([x1, y1, x2, y2, 1]) 

            
            if detections:
                track_bbs_id = tracker.update(detections)
            else:
                track_bbs_id = tracker.update(np.empty((0, 5)))
            
            saw = []
            for bbsid in track_bbs_id:
                x1, y1, x2, y2, id = bbsid
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                bottom_center = [(x1+x2)/2, y2]
                result = M @ [[bottom_center[0]], [bottom_center[1]], [1]]
                bottom_center_persp = [result[0]/result[2], result[1]/result[2]]
                saw.append(id)
                if id in db:
                    db[id].append(bottom_center_persp)
                else:
                    db[id] = []
                    db[id].append(bottom_center_persp)
                cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 255, 0))
                cv2.putText(new_frame, str(id), (x1, y1 -10), 0, 0.5, (0, 255, 255))
                if id in velocities:
                    if velocities[id] != None:
                        cv2.putText(new_frame, str(int(velocities[id] * 3.6)) + " km/h", (x1 + 25, y1 - 10), 0, 0.5, (0, 255, 255))
            
            for key, val in db.items():
                if key != 0:
                    if key not in saw:
                        db[key].append(None)
                    if len(val) > 3:
                        db[key] = db[key][1:]
                        
                    if len(db[key]) == 3:
                        if db[key][0] is not None and db[key][2] is not None:
                            vel = ((db[key][2][1] - db[key][0][1])/(3/fps))*(1/ppm)
                            velocities[key] = vel
                    # for element in db[key]:
                    #     if element is not None:
                    #         cv2.circle(new_graph, (int(element[0]), int(element[1] - car_l - 40)), 4, (0, 255, 0))
                    if db[key][-1] is not None:
                        cv2.rectangle(new_graph, (int(db[key][-1][0] - car_w/2), int(db[key][-1][1] - car_l)), (int(db[key][-1][0] + car_w/2), int(db[key][-1][1])), (0, 255, 0))
                        new_graph[int(db[key][-1][1] - car_l + 1):int(db[key][-1][1]), int(db[key][-1][0] - car_w/2 + 1):int(db[key][-1][0] + car_w/2)] = (60, 60, 60)
                        if key in velocities:
                            if velocities[key] != None:
                                cv2.putText(new_graph, str(key) + ": " + str(int(velocities[key] * 3.6)) + " km/h", (int(db[key][-1][0] - 20), int(db[key][-1][1] - car_l - 10)), 0, 0.2, (0, 255, 255))

            # print(1/(time.time() - t))
            # perspective = cv2.warpPerspective(new_frame, M, (w, h))
            # for pxl in range(0, h, 15):
            #     perspective[pxl][:] = (120, 120, 120)
            # cv2.imshow('graph', cv2.resize(graph, (int(w/1.5), int(h/1.5))))
            map = new_graph[270:, 500:700]
            map_h, map_w, map_c = map.shape
            cv2.imshow('Map', cv2.resize(map, (int(map_w*1.5), int(map_h*1.5))))
            # cv2.imshow('persp', cv2.resize(perspective, (int(w/1), int(h/1))))
            cv2.imshow('Frame', cv2.resize(new_frame, (int(w/1), int(h/1))))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_count+=1
        

    cap.release()

    cv2.destroyAllWindows()
from ultralytics import YOLO
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from sort.sort import Sort

#Initialize YOLO model and SORT tracker

model = YOLO("yolov8x.pt")
model.to('cuda')
tracker = Sort(max_age = 20) 

line_coords = [0, 450, 1366, 450]
vehicle_counter = []

filepath = "./demo.mp4"
cap = cv2.VideoCapture(filepath)
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_path, cv2.CAP_ANY, fourcc, fps, (1366, 768))

while True :

    img, frame = cap.read()
    frame = cv2.resize(frame, (1366,768), interpolation=cv2.INTER_AREA)

    detections = np.empty((0, 5))
    results = model.predict(frame, classes=[0, 1, 2, 3, 4, 5])
    
    for result in results:
        for r in result.boxes :
            x1, y1, x2, y2 = r.xyxy[0]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            score = int(r.conf[0])
            arr = np.array([x1, y1, x2, y2, score])
            detections = np.vstack((detections, arr))    


    tracker_results = tracker.update(detections)

    for track in tracker_results:
        x1, y1, x2, y2, id = track
        id = int(id)

        w = x2 - x1
        h = y2 - y1

        center_x = x1+w/2
        center_y = y1+h/2

        cv2.circle(frame, (int(center_x), int(center_y)), 2, (255,0,0), 2)
        cv2.line(frame, (0, 450), (1366, 450), (255,0,255), 2)

        if line_coords[0]<center_x<line_coords[2] and line_coords[1]<center_y<line_coords[3]+20:
            if vehicle_counter.count(id) == 0 :
                vehicle_counter.append(id)
                cv2.line(frame, (0, 450), (1366, 450), (250,0,200), 2)

        cv2.rectangle(frame,(int(x1), int(y1)), (int(x2), int(y2)), (255,0,100), 2)
        cv2.putText(frame, f"Id:{id}", (int(x1), int(y1-10)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"count:{len(vehicle_counter)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            
    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    

cap.release()
out.release()
cv2.destroyAllWindows()

    

import sys
import cv2
import threading
from datetime import datetime
from collections import defaultdict
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class VehicleTracker:
    def __init__(self, line_y=320):
        self.track_memory = {}
        self.counted_ids = set()
        self.counts = defaultdict(int)
        self.line_y = line_y
        
    def update(self, cls, cx, cy):
        label = VEHICLE_CLASSES[cls]
        obj_id = f"{label}_{cx//30}_{cy//30}"
        
        if obj_id not in self.track_memory:
            self.track_memory[obj_id] = cy
            return False
        
        prev_y = self.track_memory[obj_id]
        self.track_memory[obj_id] = cy
        
        if prev_y < self.line_y and cy >= self.line_y and obj_id not in self.counted_ids:
            self.counted_ids.add(obj_id)
            self.counts[label] += 1
            return True
        return False

class CountingThread(QThread):
    update_signal = pyqtSignal(dict, str)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, esp32_url, line_y=320):
        super().__init__()
        self.esp32_url = esp32_url
        self.line_y = line_y
        self.running = True
        
    def run(self):
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(self.esp32_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            self.update_signal.emit({}, "Cannot connect to ESP32")
            return
        
        tracker = VehicleTracker(self.line_y)
        frame_id = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = cv2.VideoCapture(self.esp32_url)
                continue
            
            frame = cv2.resize(frame, (800, 600))
            frame_id += 1
            
            if frame_id % 2 == 0:
                results = model(frame, verbose=False)[0]
                
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls not in VEHICLE_CLASSES:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    label = VEHICLE_CLASSES[cls]
                    
                    if tracker.update(cls, cx, cy):
                        msg = f"{datetime.now().strftime('%H:%M:%S')} {label} melewati garis"
                        self.update_signal.emit(tracker.counts, msg)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.line(frame, (0, self.line_y), (800, self.line_y), (0, 0, 255), 3)
            
            y = 30
            cv2.putText(frame, "VEHICLE COUNT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for vtype in ["car", "motorcycle", "bus", "truck"]:
                y += 25
                cv2.putText(frame, f"{vtype.upper()}: {tracker.counts.get(vtype, 0)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 30
            cv2.putText(frame, f"TOTAL: {sum(tracker.counts.values())}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press Q to quit", (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Vehicle Counting", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.finished_signal.emit(tracker.counts)
    
    def stop(self):
        self.running = False

class VehicleCounterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Counter")
        self.setGeometry(100, 100, 500, 500)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        layout.addWidget(QLabel("ESP32 URL:"))
        self.url_input = QLineEdit("http://10.240.28.178:81/stream")
        layout.addWidget(self.url_input)
        
        layout.addWidget(QLabel("Garis Counting (Y):"))
        self.line_input = QLineEdit("320")
        layout.addWidget(self.line_input)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_counting)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_counting)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        
        self.thread = None
        
    def start_counting(self):
        url = self.url_input.text()
        line_y = int(self.line_input.text())
        
        self.thread = CountingThread(url, line_y)
        self.thread.update_signal.connect(self.update_display)
        self.thread.finished_signal.connect(self.counting_finished)
        self.thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
    def stop_counting(self):
        if self.thread:
            self.thread.stop()
            
    def update_display(self, counts, message):
        self.log.append(message)
        self.log.append(f"Car: {counts.get('car',0)} | Motor: {counts.get('motorcycle',0)} | Bus: {counts.get('bus',0)} | Truck: {counts.get('truck',0)} | Total: {sum(counts.values())}")
        
    def counting_finished(self, counts):
        self.log.append("\nFINAL RESULT")
        self.log.append(f"Car: {counts.get('car', 0)}")
        self.log.append(f"Motorcycle: {counts.get('motorcycle', 0)}")
        self.log.append(f"Bus: {counts.get('bus', 0)}")
        self.log.append(f"Truck: {counts.get('truck', 0)}")
        self.log.append(f"Total: {sum(counts.values())}")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VehicleCounterWindow()
    window.show()
    sys.exit(app.exec())
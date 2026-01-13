import win32api
import win32con
import math
import ctypes
from ultralytics import YOLO
import dxcam
import cv2
import numpy as np
import time
import keyboard
import traceback
from collections import deque
import threading
import tkinter as tk

class autoSlicer():
    def __init__(self, visualize: bool = False,
                 SAFE_MARGIN: int = 250,
                 SWIPE_OFFSET: int = 50,
                 TIME_THRESHOLD: float = 0.125):
        '''
        Initialize and immediately start the autoSlicer app.

        Parameters
        -----------
        - visualize : bool
            Whether to visualize the detection results, by default False. Enabling this may reduce performance.
        - SAFE_MARGIN : int
            Distance from bomb, if lower than that, won't slice the fruit near it, by default 250
        - SWIPE_OFFSET : int
            The length of the swipe, lower will make a shorter swipe, by default 50
        - TIME_THRESHOLD : float
            Minimum time a fruit must be tracked before slicing (in seconds). Ts avoids slicing fruits that just appeared, by default 0.125
        -----------
        '''

        # Global Settings and Variables
        self.visualize = visualize
        self.auto_aim = False
        self.predict = True

        # For logic
        self.SAFE_MARGIN = SAFE_MARGIN
        self.SWIPE_OFFSET = SWIPE_OFFSET
        self.TIME_THRESHOLD = TIME_THRESHOLD

        self.fruit_timers = {}  # To track how long each fruit has been on screen
        self.model = YOLO("MODELS/Tuned_11n.engine", task="detect")
        self.MODEL_INPUT = 640

        # Get current screen res
        self.SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
        self.SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)
        if self.SCREEN_W > 1024 or self.SCREEN_H > 768:
            Warning("Your current input resolution is larger than 1024x768. For optimal performance, please set your screen resolution to 1024x768 or lower.")
        
        # Thread
        self.running = True # This shit will be used to stop the thread safely
        self.thread = threading.Thread(target=self.start, daemon=True)
        self.thread.start()

        self.run_ui()

    def run_ui(self):
        '''
        Ts just runs the UI gng!
        '''
        self.root = tk.Tk()
        self.root.title("Slicer Status")
        self.root.geometry("150x200")
        self.root.attributes('-topmost', True)
        self.root.configure(bg="#222")

        # Status Labels
        # Auto Aim
        self.lbl_aim = tk.Label(self.root, text="Auto Aim: OFF", font=("Arial", 10, "bold"), bg="#222", fg="red")
        self.lbl_aim.pack(pady=10)

        # Predict
        self.lbl_pred = tk.Label(self.root, text="Predict: ON", font=("Arial", 10, "bold"), bg="#222", fg="green")
        self.lbl_pred.pack(pady=10)

        # Show FPS too, i think
        self.lbl_fps = tk.Label(self.root, text="FPS: 0", font=("Arial", 10, "bold"), bg="#222", fg="green")
        self.lbl_fps.pack(pady=10)

        # Info Label
        self.lbl_info = tk.Label(self.root, text="'E': Aim\n'Z': Predict\n'Q': quit", font=("Arial", 8), bg="#222", fg="#888")
        self.lbl_info.pack(side="bottom", pady=5)

        # Disable minimize and close buttons
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.overrideredirect(True)

        self.update_ui_state()
        self.root.mainloop()

    def update_ui_state(self):
        '''
        Update the UI status labels every 100ms

        Status to show:
        - Auto Aim
        - Predict
        - FPS
        -----------
        '''
        if not self.running:
            self.root.destroy()
            return

        # Update Auto Aim
        if self.auto_aim:
            self.lbl_aim.config(text="Auto Aim: ON", fg="#00ff00")
        else:
            self.lbl_aim.config(text="Auto Aim: OFF", fg="#ff0000")

        # Update Predict
        if self.predict:
            self.lbl_pred.config(text="Predict: ON", fg="#00ff00")
        else:
            self.lbl_pred.config(text="Predict: OFF", fg="#ff0000")
        
        try:
            self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")
        except AttributeError:
            self.lbl_fps.config(text="FPS: 0")

        # Check in every 100ms
        self.root.after(100, self.update_ui_state)

    def visualizeResults(self):
        '''
        Visualize Detection Results on Screen with bounding boxes and labels.

        NOTE: This function is called only if self.visualize is True.
        '''
        frame = self.image_BGR.copy()
        if self.results is not None and len(self.results) > 0 and self.predict:
            for i, box in enumerate(self.boxes):
                cls = int(self.clss[i])
                x1, y1, x2, y2 = map(int, box)
                confidence = self.conf[i]
                id = int(self.ids[i]) if self.ids[i] is not None else "None"

                if cls == 0:    # Fruit
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"{confidence:.2f}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif cls == 1:  # Bomb
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # FPS, auto_aim and predict status
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Auto Aim: {'ON' if self.auto_aim else 'OFF'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Predict: {'ON' if self.predict else 'OFF'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("screenshot", frame)

    def logic(self):
        '''
        Core Logic for Auto Slicing.
        - For each detected fruit, check if it's been on screen longer than TIME_THRESHOLD.
        - Check distance to all detected bombs, if within SAFE_MARGIN, skip slicing.
        - If safe, perform swipe action towards the fruit.
        '''

        if self.results is None or len(self.results) == 0:
            return        
        if not hasattr(self, 'ids') or self.ids is None:
            return
        
        current_time = time.time()
        current_frame_ids = set() 

        fruits = [] 
        bombs = [] 

        # Get detected objects
        for i, cls in enumerate(self.clss):
            box = self.boxes[i]
            current_id = int(self.ids[i]) if self.ids[i] is not None else None

            # Get center coordinates
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            
            if int(cls) == 0: # Fruit
                if current_id is not None:
                    current_frame_ids.add(current_id)       # Track on current frame IDs
                    if current_id not in self.fruit_timers:
                        self.fruit_timers[current_id] = current_time
                    fruits.append((cx, cy, current_id))

            elif int(cls) == 1: # Bomb
                bombs.append((cx, cy))
        
        # Cleanup Timers for fruits no longer detected
        for fid in list(self.fruit_timers.keys()):
            if fid not in current_frame_ids:
                del self.fruit_timers[fid]

        # Execution
        for (fx, fy, fid) in fruits:

            # Time Check
            time_on_screen = current_time - self.fruit_timers.get(fid, current_time)
            if time_on_screen < self.TIME_THRESHOLD:
                continue 

            # Bomb Check
            is_safe = True
            for (bx, by) in bombs:
                distance = math.sqrt((fx - bx)**2 + (fy - by)**2)
                if distance < self.SAFE_MARGIN:
                    is_safe = False
                    break 
            
            if is_safe:
                self.perform_slice(fx, fy)
                time.sleep(0.01)
                
                # Remove from timer so we don't spam-slice the same fruit
                if fid in self.fruit_timers:
                    del self.fruit_timers[fid]

    def perform_slice(self, target_x, target_y):
        '''
        Perform a swipe action towards the target coordinates from (start_x, start_y) to (end_x, end_y).

        Parameters
        -----------
        - target_x : int
            The x-coordinate of the target center point.
        - target_y : int
            The y-coordinate of the target center point.
        -----------
        '''
        start_x = int(target_x - self.SWIPE_OFFSET)
        start_y = int(target_y + self.SWIPE_OFFSET)
        end_x   = int(target_x + self.SWIPE_OFFSET)
        end_y   = int(target_y - self.SWIPE_OFFSET)

        win32api.SetCursorPos((start_x, start_y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0) # Press down after positioning on (start_x, start_y)
        time.sleep(0.015)  
    
        win32api.SetCursorPos((end_x, end_y)) # Drag to (end_x, end_y)
        time.sleep(0.015)
        
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0) # Release the mouse button

    def predictFrame(self):
        '''
        Predict on the current frame and update detection results.
        '''
        results = self.model.track(self.image_BGR, persist=True, verbose=False ,
                            tracker="CustomTracker.yaml",
                            conf=0.1, iou=0.5,
                            half=True)
        self.results = results[0].boxes

        # Unwrap contents if there are results
        if self.results is not None and len(self.results) > 0:
            self.ids = self.results.id.cpu().numpy()  if self.results.id is not None else [None]*len(self.results)
            self.boxes = self.results.xyxy.cpu().numpy()
            self.clss = self.results.cls.cpu().numpy()
            self.conf = self.results.conf.cpu().numpy()

    def start(self):
        '''
        Backend Thread to capture screen, predict and execute logic.

        Controls:
        - 'E': Toggle Auto Aim
        - 'Z': Toggle Predict
        - 'Q': Quit the script
        -----------
        '''
        frame_count = 0
        time_que = deque(maxlen=60)
        screen = dxcam.create(device_idx=0, output_idx=0, output_color="BGR")
        screen.start(target_fps=100)

        try:
            while True:
                frame_count += 1
                if keyboard.is_pressed("e") and frame_count % 10 == 0:
                    self.auto_aim = not self.auto_aim
                if keyboard.is_pressed("z") and frame_count % 10 == 0:
                    self.predict = not self.predict
                if keyboard.is_pressed("q") and frame_count % 10 == 0:
                    break

                time_st = time.time()
                self.image_BGR = screen.get_latest_frame()

                if self.predict:
                    self.predictFrame()

                if self.auto_aim:
                    self.logic()

                # FPS Calculation
                time_et = time.time()
                elapsed = time_et - time_st
                self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
                time_que.append(self.fps)
                
                if self.visualize:
                    self.visualizeResults()
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

        finally:
            avg_fps = np.mean(time_que) if time_que else 0.0
            print("Average FPS: {:.2f}".format(avg_fps))

            # Safe Exit, close CV2, stop screen capture, and end thread
            cv2.destroyAllWindows()
            screen.stop()
            self.running = False

autoSlicer(visualize=True, SAFE_MARGIN=200, SWIPE_OFFSET=50, TIME_THRESHOLD=0.125)
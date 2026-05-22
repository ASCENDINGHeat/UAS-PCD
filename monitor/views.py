import os
import cv2
import time
import threading
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO

def index(request):
    """Renders the diagnostic dashboard container."""
    return render(request, 'index.html')


class AsynchronousCrackTracker:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.raw_frame = None
        self.thresh_frame = None
        self.is_running = True
        
        self.lock = threading.Lock()
        self.model = YOLO('best2.onnx', task='detect')
        
        threading.Thread(target=self._video_grabber_loop, daemon=True).start()
        threading.Thread(target=self._crack_processor_loop, daemon=True).start()

    def _video_grabber_loop(self):
        """Thread 1: Safely samples frames via a decoupled thread pointer."""
        while self.is_running:
            with self.lock:
                local_cap = self.cap
            
            if local_cap is not None and local_cap.isOpened():
                ret, frame = local_cap.read()
                if ret:
                    with self.lock:
                        # Ensure the capture source hasn't switched mid-read
                        if self.cap == local_cap:
                            self.raw_frame = frame
            time.sleep(0.01)

    def _crack_processor_loop(self):
        """Thread 2: Localized YOLO region inference loop with Geometry Filtering."""
        target_fps = 10
        interval = 1.0 / target_fps
        while self.is_running:
            start_time = time.time()
            local_frame = None
            with self.lock:
                if self.raw_frame is not None:
                    local_frame = self.raw_frame.copy()
            
            if local_frame is not None:
                h, w = local_frame.shape[:2]
                composite_mask = np.zeros((h, w), dtype=np.uint8)
                results = self.model(local_frame, verbose=False, device='cpu')[0]
                
                if results.boxes is not None:
                    for box in results.boxes:
                        if float(box.conf[0]) < 0.25: continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if (x2 - x1) < 5 or (y2 - y1) < 5: continue
                        
                        # 1. Extract the localized crop area
                        crop = local_frame[y1:y2, x1:x2]
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
                        
                        # 2. Initial noisy raw threshold pass
                        thresh = cv2.adaptiveThreshold(
                            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY_INV, blockSize=11, C=3
                        )
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                        
                        # 3. GEOMETRY FILTER: Analyze individual shapes inside this specific crop
                        clean_crop_mask = np.zeros_like(closed)
                        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area < 15:  # Instant drop for micro-noise dots
                                continue
                                
                            # Calculate structural shape metrics
                            _, _, w_c, h_c = cv2.boundingRect(cnt)
                            perimeter = cv2.arcLength(cnt, True)
                            
                            # Elongation ratio (how stretched out the contour shape is)
                            aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                            
                            # Core Logic: Keep it if it is thin/long OR forms a substantial line segment
                            if aspect_ratio > 2.5 or perimeter > 80 or area > 300:
                                cv2.drawContours(clean_crop_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                        
                        # 4. Splice ONLY the verified geometric crack shapes back onto the main canvas
                        composite_mask[y1:y2, x1:x2] = cv2.bitwise_or(composite_mask[y1:y2, x1:x2], clean_crop_mask)
                
                with self.lock:
                    self.thresh_frame = composite_mask
            
            elapsed = time.time() - start_time
            time.sleep(max(0, interval - elapsed))

    def switch_source(self, new_source):
        """
        Pre-warms the new camera device/URL in the background outside the lock.
        Prevents thread deadlocks if network links lag or time out.
        """
        # Convert string to integer index if it's a local hardware port flag
        if str(new_source).isdigit():
            new_source = int(new_source)
            
        print(f"[TRACKER] Initializing new stream target target: {new_source}")
        temp_cap = cv2.VideoCapture(new_source)
        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Swap streams instantly once initialized
        with self.lock:
            old_cap = self.cap
            self.cap = temp_cap
            # Reset existing image matrix memory structures to clear visual artifacts
            self.raw_frame = None
            self.thresh_frame = None
            
        if old_cap is not None:
            old_cap.release()
        print("[TRACKER] Stream target successfully swapped.")

    def get_encoded_frame(self):
        with self.lock:
            if self.raw_frame is None: return None
            output = self.raw_frame.copy()
            if self.thresh_frame is not None and self.thresh_frame.shape[:2] == output.shape[:2]:
                output[self.thresh_frame == 255] = [0, 0, 255]
        ret, jpeg = cv2.imencode('.jpg', output)
        return jpeg.tobytes() if ret else None


# Global tracker engine resource instantiation
tracker = AsynchronousCrackTracker(0)

def gen_frames():
    while True:
        frame_bytes = tracker.get_encoded_frame()
        if frame_bytes is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        time.sleep(1.0 / 30.0)

def video_feed_blended(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def change_source(request):
    """API Endpoint to catch selection requests without breaking continuous streaming."""
    if request.method == 'POST':
        selected_source = request.POST.get('source', '0')
        
        # Spawn a non-blocking thread to handle the pre-warm sequence
        threading.Thread(target=tracker.switch_source, args=(selected_source,), daemon=True).start()
        
        return JsonResponse({'status': 'success', 'active_source': selected_source})
    return JsonResponse({'status': 'error', 'message': 'Invalid Method'}, status=400)
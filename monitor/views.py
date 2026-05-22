import os
import cv2
import time
import threading
import numpy as np
import base64
import uuid
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
        self.is_paused = False
        
        self.lock = threading.Lock()
        self.model = YOLO('best2.onnx', task='detect')
        
        threading.Thread(target=self._video_grabber_loop, daemon=True).start()
        threading.Thread(target=self._crack_processor_loop, daemon=True).start()

    def _video_grabber_loop(self):
        """Thread 1: Safely samples frames via a decoupled thread pointer."""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
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
        target_fps = 5
        interval = 1.0 / target_fps
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
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

    def get_processed_live_frame(self, mode):
        with self.lock:
            if self.raw_frame is None: return None
            output = self.raw_frame.copy()
            
        if mode == 'grayscale':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            ret, jpeg = cv2.imencode('.jpg', gray)
            return jpeg.tobytes() if ret else None
            
        elif mode == 'denoised':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            # Use GaussianBlur for fast real-time 30 FPS processing
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, jpeg = cv2.imencode('.jpg', denoised)
            return jpeg.tobytes() if ret else None
            
        elif mode == 'thresholded':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, blockSize=11, C=3
            )
            ret, jpeg = cv2.imencode('.jpg', thresh)
            return jpeg.tobytes() if ret else None
            
        elif mode == 'morphological':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, blockSize=11, C=3
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            ret, jpeg = cv2.imencode('.jpg', closed)
            return jpeg.tobytes() if ret else None
            
        return None


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

def gen_frames_processed(mode):
    while True:
        frame_bytes = tracker.get_processed_live_frame(mode)
        if frame_bytes is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        time.sleep(1.0 / 30.0)

def video_feed_blended(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed_grayscale(request):
    return StreamingHttpResponse(gen_frames_processed('grayscale'), content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed_denoised(request):
    return StreamingHttpResponse(gen_frames_processed('denoised'), content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed_thresholded(request):
    return StreamingHttpResponse(gen_frames_processed('thresholded'), content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed_morphological(request):
    return StreamingHttpResponse(gen_frames_processed('morphological'), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def change_source(request):
    """API Endpoint to catch selection requests without breaking continuous streaming."""
    if request.method == 'POST':
        selected_source = request.POST.get('source', '0')
        
        # Spawn a non-blocking thread to handle the pre-warm sequence
        threading.Thread(target=tracker.switch_source, args=(selected_source,), daemon=True).start()
        
        return JsonResponse({'status': 'success', 'active_source': selected_source})
    return JsonResponse({'status': 'error', 'message': 'Invalid Method'}, status=400)


captured_buffer = {}


@csrf_exempt
def toggle_stream(request):
    """API Endpoint to pause/resume the camera grabber and processing threads."""
    if request.method == 'POST':
        paused_str = request.POST.get('paused', 'false').lower()
        paused = paused_str == 'true'
        tracker.is_paused = paused
        print(f"[TRACKER] Stream active pause status toggled to: {tracker.is_paused}")
        return JsonResponse({'status': 'success', 'is_paused': tracker.is_paused})
    return JsonResponse({'status': 'error', 'message': 'Invalid Method'}, status=400)


@csrf_exempt
def capture_still(request):
    """API Endpoint to capture the current frame and store it in-memory."""
    if request.method == 'POST' or request.method == 'GET':
        with tracker.lock:
            if tracker.raw_frame is None:
                return JsonResponse({'status': 'error', 'message': 'No active frame captured yet.'}, status=400)
            frame_to_store = tracker.raw_frame.copy()
            
        still_id = str(uuid.uuid4())
        captured_buffer[still_id] = frame_to_store
        
        # Return dimensions and success
        h, w = frame_to_store.shape[:2]
        return JsonResponse({
            'status': 'success',
            'still_id': still_id,
            'width': w,
            'height': h
        })
    return JsonResponse({'status': 'error', 'message': 'Invalid Method'}, status=400)


@csrf_exempt
def upload_image(request):
    """API Endpoint to upload a local image file and cache it in captured_buffer."""
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': 'No image file provided in upload.'}, status=400)
            
        uploaded_file = request.FILES['image']
        try:
            # Read file data directly from memory into numpy array
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return JsonResponse({'status': 'error', 'message': 'Failed to decode image file format.'}, status=400)
                
            h, w = img.shape[:2]
            # Store in captured_buffer
            still_id = str(uuid.uuid4())
            captured_buffer[still_id] = img
            
            print(f"[TRACKER] Local image uploaded successfully. UUID: {still_id}, resolution: {w}x{h}")
            return JsonResponse({
                'status': 'success', 
                'still_id': still_id,
                'width': w,
                'height': h
            })
        except Exception as e:
            print(f"[ERROR] Error processing uploaded file: {str(e)}")
            return JsonResponse({'status': 'error', 'message': f'Internal error processing file: {str(e)}'}, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'Invalid Method'}, status=400)


@csrf_exempt
def analyze_captured_frame(request):
    """API Endpoint to perform parameterized crack analysis on a stored frame."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST is allowed'}, status=400)
        
    still_id = request.POST.get('still_id')
    if not still_id or still_id not in captured_buffer:
        return JsonResponse({'status': 'error', 'message': 'Valid still_id is required.'}, status=400)
        
    raw_frame = captured_buffer[still_id]
    
    # Parse parameters
    try:
        brightness = int(request.POST.get('brightness', 0))
        contrast = int(request.POST.get('contrast', 0))
        auto_enhance = request.POST.get('auto_enhance', 'false').lower() == 'true'
        bilateral_d = int(request.POST.get('bilateral_d', 5))
        bilateral_sigma_color = int(request.POST.get('bilateral_sigma_color', 40))
        bilateral_sigma_space = int(request.POST.get('bilateral_sigma_space', 40))
        threshold_block_size = int(request.POST.get('threshold_block_size', 11))
        threshold_c = int(request.POST.get('threshold_c', 3))
        min_area = int(request.POST.get('min_area', 15))
        min_perimeter = int(request.POST.get('min_perimeter', 80))
        min_aspect_ratio = float(request.POST.get('min_aspect_ratio', 2.5))
    except ValueError as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid parameter values: {str(e)}'}, status=400)
        
    # Ensure block size is odd
    if threshold_block_size % 2 == 0:
        threshold_block_size += 1
        
    # 1. Adjust brightness & contrast
    alpha = 1.0 + (contrast / 100.0) if contrast >= 0 else 1.0 + (contrast / 130.0)
    beta = float(brightness)
    processed_frame = cv2.convertScaleAbs(raw_frame, alpha=alpha, beta=beta)
    
    # 2. Automated Enhancement (CLAHE)
    if auto_enhance:
        ycrcb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(ycrcb))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        channels[0] = clahe.apply(channels[0])
        ycrcb = cv2.merge(channels)
        processed_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
    # 3. Model Inference
    results = tracker.model(processed_frame, verbose=False, device='cpu')[0]
    
    h, w = processed_frame.shape[:2]
    composite_mask = np.zeros((h, w), dtype=np.uint8)
    blended_overlay = processed_frame.copy()
    
    crack_reports = []
    crack_counter = 0
    
    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            if (bx2 - bx1) < 5 or (by2 - by1) < 5:
                continue
                
            # Perform local contour extraction inside bounding box crop
            crop = processed_frame[by1:by2, bx1:bx2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.bilateralFilter(
                gray, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space
            )
            
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=threshold_block_size, C=threshold_c
            )
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            clean_crop_mask = np.zeros_like(closed)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                    
                _, _, wc, hc = cv2.boundingRect(cnt)
                perimeter = cv2.arcLength(cnt, True)
                aspect_ratio = max(wc, hc) / min(wc, hc) if min(wc, hc) > 0 else 1
                
                # Filter cracks based on geometry criteria
                if aspect_ratio >= min_aspect_ratio or perimeter >= min_perimeter or area >= 300:
                    cv2.drawContours(clean_crop_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    
                    crack_counter += 1
                    
                    # Compute global coordinates of bounding box of this contour
                    cnt_x = cnt[:, :, 0]
                    cnt_y = cnt[:, :, 1]
                    gx1 = bx1 + int(np.min(cnt_x))
                    gy1 = by1 + int(np.min(cnt_y))
                    gx2 = bx1 + int(np.max(cnt_x))
                    gy2 = by1 + int(np.max(cnt_y))
                    
                    severity = "Minor"
                    if perimeter > 120 or area > 400:
                        severity = "Critical"
                    elif perimeter > 50 or area > 120:
                        severity = "Moderate"
                        
                    crack_reports.append({
                        'id': crack_counter,
                        'box': [gx1, gy1, gx2, gy2],
                        'width_px': int(wc),
                        'height_px': int(hc),
                        'perimeter_px': round(float(perimeter), 2),
                        'area_px': round(float(area), 2),
                        'elongation': round(float(aspect_ratio), 2),
                        'severity': severity
                    })
            
            # Splice clean crop mask back to composite mask
            composite_mask[by1:by2, bx1:bx2] = cv2.bitwise_or(composite_mask[by1:by2, bx1:bx2], clean_crop_mask)
            
            # Draw Region Bounding Box (Cyan) on overlay
            cv2.rectangle(blended_overlay, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
            cv2.putText(
                blended_overlay, f"Region {conf*100:.0f}%", 
                (bx1, by1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA
            )
            
    # Apply Pink Overlay on blended_overlay where cracks are found
    blended_overlay[composite_mask == 255] = [255, 0, 255]
    
    # Helper to convert to base64
    def to_b64(img):
        ret, jpeg = cv2.imencode('.jpg', img)
        if ret:
            return base64.b64encode(jpeg.tobytes()).decode('utf-8')
        return ""
        
    raw_b64 = to_b64(processed_frame)
    mask_b64 = to_b64(composite_mask)
    blended_b64 = to_b64(blended_overlay)
    
    return JsonResponse({
        'status': 'success',
        'raw_image': raw_b64,
        'mask_image': mask_b64,
        'blended_image': blended_b64,
        'cracks': crack_reports,
        'total_cracks': len(crack_reports),
        'critical_count': sum(1 for c in crack_reports if c['severity'] == 'Critical'),
        'moderate_count': sum(1 for c in crack_reports if c['severity'] == 'Moderate')
    })
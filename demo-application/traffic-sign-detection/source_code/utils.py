import os
import sys
import numpy as np
import cv2 
from cv2 import VideoCapture, VideoWriter

# bounding box: (x, y, w, h), shape_type, apply for bounding box set
BoundingBox = tuple[tuple[int, int, int, int], str]
# region: roi, (x, y, w, h), shape_type, region area for detected object
Region = tuple[np.ndarray, tuple[int, int, int, int], str]

class FrameProcessor():
    """
    This class processes individual video frames to detect and recognize traffic signs.
    It includes methods for preprocessing, color masking, ROIs extraction and template matching.
    __call__:   - method processes a frame and returns the annotated frame.
                - input: frame (np.ndarray): The input video frame.
                         frame_idx (int): The index of the current frame.
                - output: frame with bounding boxes and labels.
    """
    # 1. Preprocess
    def preprocess(
        self, frame: np.ndarray, 
        gamma: float = 1.8, 
        blur_size: tuple[int, int] = (5,5), 
        blur_kernel: int = 0,
        hsv_type: int = cv2.COLOR_BGR2HSV
    ) -> np.ndarray:
        # adjust gamma, make each pixels nearly together
        def adjust_gamma(frame: np.ndarray, gamma: float):
            invGamma = 1 / gamma
            table = np.array([((i / 255) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
            return cv2.LUT(frame, table)
        
        gm_frame = adjust_gamma(frame, gamma=gamma)
        smooth_frame = cv2.GaussianBlur(gm_frame, blur_size, blur_kernel) # Gaussian blur to reduce noise
        hsv_frame = cv2.cvtColor(smooth_frame, hsv_type) # ensure input of next step is HSV
        
        return hsv_frame
    
    # 2. Masks for color detection
    
    def masks_color(self, preprocessed_frame: np.ndarray) -> np.ndarray:
        # Create color masks for red, blue, and yellow colors in the HSV color space
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 220, 220])
        lower_red2 = np.array([120, 20, 40])
        upper_red2 = np.array([179, 255, 255])
        lower_blue = np.array([100, 95, 85])
        upper_blue = np.array([130, 255, 255])
        lower_yellow1 = np.array([8, 95, 100])
        upper_yellow1 = np.array([33, 255, 255])

        mask_red = cv2.bitwise_or(
            cv2.inRange(preprocessed_frame, lower_red1, upper_red1),
            cv2.inRange(preprocessed_frame, lower_red2, upper_red2)
        )
        mask_blue = cv2.inRange(preprocessed_frame, lower_blue, upper_blue)
        
        mask_yellow = cv2.inRange(preprocessed_frame, lower_yellow1, upper_yellow1)
        
        mask_combine = cv2.bitwise_or(mask_red, mask_blue)
        mask_combine = cv2.bitwise_or(mask_combine, mask_yellow)
        
        # Apply morphological opening to clean up the mask
        kernel_open = np.ones((3, 3), np.uint8)
        mask_combine = cv2.morphologyEx(mask_combine, cv2.MORPH_OPEN, kernel_open)
        
        return mask_combine
    
    # 3. Finding regions of interest (ROIs)
    
    def bounding_boxes(self, mask: np.ndarray, frame: np.ndarray) -> set[BoundingBox]:
        # Detect bounding boxes and classify shapes (circle, triangle, rectangle)
        bounding_boxes: set[BoundingBox] = set()
        # auto contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 600 or area > 50000:
                continue
            
            # rectangle detection
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            
            # check triangle if round shape
            if len(approx) == 3:
                pts = [approx[i][0] for i in range(3)]
                d1 = np.linalg.norm(pts[0] - pts[1])
                d2 = np.linalg.norm(pts[1] - pts[2])
                d3 = np.linalg.norm(pts[2] - pts[0])

                average = (d1 + d2 + d3) / 3
                deviation = 0.3 * average # 30% nearly round shape

                if abs(d1 - average) < deviation and abs(d2 - average) < deviation and abs(d3 - average) < deviation:
                    bounding_boxes.add((cv2.boundingRect(approx), "triangle"))
                    continue
            
            # last case, circle detection by similar
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            circle_area = np.pi * (radius ** 2)
            circularity = area / circle_area
            if circularity > 0.67:
                box = (center[0] - radius, center[1] - radius, radius * 2, radius * 2)
                bounding_boxes.add((box, "circle")) 
                
        return bounding_boxes
    
    def valid_bounding_box(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = box
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        return x, y, w, h
    
    def roi_frame(self, masks: np.ndarray, frame: np.ndarray, frame_idx: int
                  ) -> list[Region]:
        # cut regions of interest from the frame based on bounding boxes
        bounding_boxes = [(self.valid_bounding_box(box), box_type) for box, box_type in self.bounding_boxes(masks, frame)]
        regions: list[Region] = []
        
        for box, type in bounding_boxes:
            x, y, w, h = box
            roi = frame[y:y+h, x:x+w]
            regions.append((roi, box, type))
            
        return regions
    
    def get_template_paths(self, type: str) -> list[np.ndarray]:
        if type == "circle":
            return {"label/template_image/phai_di_vong_sang_phai.png" : 6,
                    "label/template_image/cam_di_nguoc_chieu.png" : 1,
                    "label/template_image/cam_do_xe.png" : 2,
                    "label/template_image/cam_dung_xe_va_do_xe.png" : 3,
                    "label/template_image/cam_re_trai.png" : 4}
            
        if type == "triangle":
            return {"label/template_image/di_cham.png" : 5,
                    "label/template_image/tre_em.png" : 9}
        
        return {"label/template_image/chi_huong_duong_1.png" : 7,
                "label/template_image/chi_huong_duong_2.png" : 8,
                "label/template_image/thong_bao_tai_nan.png" : 10}
    
    def resize_template(self, template_path: str, roi: np.ndarray, resize_ratio: float) -> np.ndarray:
        # preprocess template and resize to fit roi
        template = cv2.imread(template_path)
        roi_h, roi_w = roi.shape[:2]
        tpl_h, tpl_w = template.shape[:2]
        base_scale = min(roi_w / tpl_w, roi_h / tpl_h)
        scale = max(base_scale * resize_ratio, 1e-3)

        new_w = max(1, int(round(tpl_w * scale)))
        new_h = max(1, int(round(tpl_h * scale)))

        if new_w == tpl_w and new_h == tpl_h:
            return template.copy()

        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized_template = cv2.resize(template, (new_w, new_h), interpolation=interpolation)
        return resized_template
    
    # 4. Sign recognition using template matching
    
    def template_matching(self, resize_template: np.ndarray, roi: np.ndarray, threshold: float, label: int) -> float:
        """
        Just template matching function to return max_val if greater than threshold
        """
        res = cv2.matchTemplate(roi, resize_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return max_val
        return 0
    
    def detected_label(self, roi: np.ndarray, type: str) -> int:
        template_paths = self.get_template_paths(type)
        max_label = -1
        max_val = 0
        for template_path, label in template_paths.items():
            # try different resize ratios for better matching
            # 1 -> threshold 0.25 because this case for the big area detected, hardly to fit
            resized_template_1 = self.resize_template(template_path, roi, resize_ratio=0.9)
            resized_template_2 = self.resize_template(template_path, roi, resize_ratio=1)
            resized_template_3 = self.resize_template(template_path, roi, resize_ratio=0.8)
            val_temp_1 = self.template_matching(resized_template_1, roi, threshold=0, label=label)
            val_temp_2 = self.template_matching(resized_template_2, roi, threshold=0, label=label)
            val_temp_3 = self.template_matching(resized_template_3, roi, threshold=0, label=label)
            if val_temp_1 > max_val:
                max_val = val_temp_1
                max_label = label
            if val_temp_2 > max_val:
                max_val = val_temp_2
                max_label = label
            if val_temp_3 > max_val:
                max_val = val_temp_3
                max_label = label
        return max_label, max_val
    
    def detected_value(self, label: int) -> str:
        label_dict = {
            1: "CAM DI NGUOC CHIEU",
            2: "CAM DO XE",
            3: "CAM DUNG XE VA DO XE",
            4: "CAM RE TRAI",
            5: "DI CHAM",
            6: "PHAI DI VONG SANG PHAI",
            0: "TRE EM HAY BANG QUA DUONG",
        }
        return label_dict.get(label, "Unknown")
    
    def draw_bounding_boxes(self, roi_frame: tuple[np.ndarray, int, int, int, int, str],
                            frame: np.ndarray):
        """Return first detected ROI with its label info."""
        frame_info = []
        for roi, box, type in roi_frame:
            label, value = self.detected_label(roi, type)
            if label == -1:
                continue
            if value > 0.3:
                frame_info.append((label, box, roi, True))
            else:
                frame_info.append((label, box, roi, False))
        return frame_info
            
    
    def __call__(self, frame: np.ndarray, frame_idx: int):
        preprocessed_frame = self.preprocess(frame)
        masks = self.masks_color(preprocessed_frame)
        roi_frame = self.roi_frame(masks, frame, frame_idx)
        return self.draw_bounding_boxes(roi_frame, frame)

class VideoProcessor():
    """
    This class use for processing video files. Read video then apply frame processing.
    
    """
    def __init__(self, input_path: str) -> None:
        self.cap = VideoCapture(input_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        self.src = input_path
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def get_info(self) -> dict:
        # use for getting video information
        return {
            "src": self.src,
            "size": self.size,
            "fps": self.fps,
            "total_frames": self.total_frames
        }
        
    def video_process(self, frame_process: FrameProcessor) -> None:
        """Process video frame-by-frame, save ROIs, and log frames with no detections."""
        # Create directories if they do not exist
        os.makedirs("datasets/images/train", exist_ok=True)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # make train_data
            # Save frame image
            cv2.imwrite(f"datasets/images/train/frame_{frame_idx:04d}.png", frame)
            
            # ROIs and labeling then human check to remove false positives
            frame_info = frame_process(frame, frame_idx)
            has_roi = True if frame_info != [] else False

            if has_roi:
                for detected in frame_info:
                    label, box, roi, detected = detected
                    label_dir = str(label) if label != -1 else "unknown"
                    target_root = "label/true_label" if detected else "label/false_label"
                    save_dir = os.path.join(target_root, label_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    filename = f"{frame_idx}_{box[0]}_{box[1]}_{box[2]}_{box[3]}.png"
                    cv2.imwrite(os.path.join(save_dir, filename), roi)
            else:
                with open("label/frame_no_detected.txt", "a", encoding="utf-8") as f:
                    f.write(f"{frame_idx}\n")

            frame_idx += 1
            if self.total_frames:
                percent = (frame_idx / self.total_frames) * 100
                percent_int = min(int(percent/2), 50)
                bar = "=" * percent_int
                print(f"\r[INFO] Processing: {bar}{percent:5.1f}% ({frame_idx}/{self.total_frames})", end='', flush=True)
            else:
                print(f"\r[INFO] Processed {frame_idx} frames", end='', flush=True)

        self.cap.release()
        print()
        
    def open_save_video(self, output_path: str) -> None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use mp4v to avoid  XVID->mp4
        self.out = VideoWriter(output_path, fourcc, self.fps, self.size)
        if not self.out.isOpened():
            raise RuntimeError("Cannot open VideoWriter. Check codec/output path.")
        
    def close_save_video(self) -> None:
        self.cap.release()
        self.out.release()
        
    @staticmethod
    def labels(input_path: str, output_path: str, width: int, height: int) -> None:
        print(f"[INFO] Generating labels in YOLO format from {input_path} to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        for i in range(3087):
            with open(f"{output_path}/frame_{i:04d}.txt", "w") as f:
                pass
        for folder_name in os.listdir(input_path):
            for file_name in os.listdir(os.path.join(input_path, folder_name)):
                classifirer = int(folder_name) - 1
                frame_idx_str, x_str, y_str, w_str, h_str = file_name.rstrip('.png').split('_')
                x, y, w, h = int(x_str), int(y_str), int(w_str), int(h_str)
                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                norm_w = w / width
                norm_h = h / height
                frame_idx = int(frame_idx_str)
                with open (f"{output_path}/frame_{frame_idx:04d}.txt", "a") as f:
                    f.write(f"{classifirer} {x_center} {y_center} {norm_w} {norm_h}\n")
                    
        print("[INFO] Label generation completed.")
        
        
   
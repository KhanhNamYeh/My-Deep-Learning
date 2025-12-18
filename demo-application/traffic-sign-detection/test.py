from ultralytics import YOLO
import cv2
from cv2 import VideoCapture, VideoWriter
import os

# Đường dẫn weights bạn đã train
WEIGHTS = "runs/detect/traffic_detect_run3/weights/best.pt"  # đổi nếu khác

model = YOLO(WEIGHTS)



def draw_box(frame, box, cls_name: str, score: float):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, cls_name, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"{score:.2f}", (x - 50, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


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
        
    def video_process(self, model: YOLO, output_path: str) -> None:
        # process video frame by frame
        self.open_save_video(output_path)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            result = model(frame)[0]
            for xyxy, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = xyxy.tolist()
                box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                cls_name = model.names[int(cls_id)]
                draw_box(frame, box, cls_name, float(conf))

            self.out.write(frame)
            
            frame_idx += 1
            # Progress display
            if self.total_frames:
                percent = (frame_idx / self.total_frames) * 100
                percent_int = min(int(percent/2), 50)
                bar = "=" * percent_int
                print(f"\r[INFO] Processing: {bar}{percent:5.1f}% ({frame_idx}/{self.total_frames})", end='', flush=True)
            else:
                print(f"\r[INFO] Processed {frame_idx} frames", end='', flush=True)
        self.close_save_video()
        print()

if __name__ == "__main__":
    input_video_path = "video/train_video.mp4"
    output_video_path = "video/output_video.mp4"
    processor = VideoProcessor(input_video_path)
    info = processor.get_info()
    processor.video_process(model, output_video_path)
    print("[INFO] Video processing completed.")
from ultralytics import YOLO
import cv2
from auto_label import VideoProcessor
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
    
VideoProcessor.video_process_w_yolo = video_process

if __name__ == "__main__":
    input_video_path = "video/train_video.mp4"
    output_video_path = "video/output_video.mp4"
    processor = VideoProcessor(input_video_path)
    info = processor.get_info()
    processor.video_process_w_yolo(model, output_video_path)
    print("[INFO] Video processing completed.")
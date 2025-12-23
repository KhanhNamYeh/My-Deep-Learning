from ultralytics import YOLO

def train_model():
    model = YOLO('yolo11n.pt') 

    results = model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640, 
        device=0,
        batch=16,
        workers=0,
        name='traffic_detect_run' # Tên folder lưu kết quả
    )

def resume_training():
    # QUAN TRỌNG: Hãy kiểm tra đường dẫn tới file last.pt của bạn.
    # Dựa trên code cũ của bạn (name='traffic_detect_run'), đường dẫn thường sẽ là:
    # 'runs/detect/traffic_detect_run/weights/last.pt'
    # Nếu bạn đã chạy nhiều lần, folder có thể là traffic_detect_run2, traffic_detect_run3...
    path_to_last_checkpoint = 'runs/detect/traffic_detect_run3/weights/last.pt'

    try:
        # 1. Load weights từ epoch gần nhất
        model = YOLO(path_to_last_checkpoint) 

        # 2. Tiếp tục train
        # Khi resume=True, nó tự động lấy lại các setting cũ (epochs, data, batch...) 
        # đã lưu trong file .pt, bạn không cần khai báo lại.
        results = model.train(resume=True)
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {path_to_last_checkpoint}")
        print("Vui lòng kiểm tra lại folder 'runs/detect/...' để tìm đúng file last.pt")

if __name__ == '__main__':
    resume_training()
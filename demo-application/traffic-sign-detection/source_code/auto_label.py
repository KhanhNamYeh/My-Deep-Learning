from utils import VideoProcessor, FrameProcessor
import sys 
 
def main(argv):   
    INPUT_VIDEO = "video/train_video.mp4"
    input_video = INPUT_VIDEO
    video_processor = VideoProcessor(input_video)
    frame_processor = FrameProcessor()
    info = video_processor.get_info()
    print("[INFO] Video details:")
    print(f"      -Source       : {info['src']}")
    print(f"      -Resolution   : {info['size'][0]}x{info['size'][1]}")
    print(f"      -Frame rate   : {info['fps']:.2f} fps")
    print(f"      -Total frames : {info['total_frames']}")
    print("[INFO] Starting video processing...")
    if len(argv) == 2 and argv[1] == "labeling":
        video_processor.video_process(frame_processor)
    if len(argv) == 1 or argv[1] == "transform":
        video_processor.labels("label/true_label", "datasets/labels/train", info['size'][0], info['size'][1])
if __name__ == "__main__":
    main(sys.argv)
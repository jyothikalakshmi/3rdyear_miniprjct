import cv2
import os
import shutil
import matplotlib.pyplot as plt

# Set the path to your video file (replace this with your actual video file path)
# video_path = 'C:\\Users\\Admin\\Desktop\\miniprjct_3rd_year\\3rdyear_miniprjct\\videos\\road_traffic.mp4'  # Change this path
# video_path = 'C:/Users/Admin/Desktop/miniprjct_3rd_year/3rdyear_miniprjct/videos/road_traffic.mp4'
video_path = r"C:\Users\Admin\Desktop\miniprjct_3rd_year\3rdyear_miniprjct\videos\road_trafifc.mp4"


# Run YOLOv5 detection on the video
os.system(f'python detect.py --source {video_path} --weights yolov5s.pt --conf 0.3 --line-thickness 1 --project runs/detect --name exp_video1')

# Path to the processed video in the output directory
output_video_path = 'runs/detect/exp_video/road_traffic.mp4'  # Change to match the actual output path

# Check if the video exists
if os.path.exists(output_video_path):
    print(f"Processed video saved at: {output_video_path}")
    # Optionally, move the video to a desired location (e.g., outside the 'runs' folder)
    shutil.move(output_video_path, 'test_video_processed.mp4')  # Change the destination path as needed
else:
    print("Processed video not found. Please check the YOLOv5 output.")
    
# Optionally, display the processed video (if you want to check it in Jupyter or IPython)
# from IPython.display import Video
# Video(output_video_path)





import cv2
import matplotlib.pyplot as plt
import os

# image_path = '..\images\obj.jpg'  # Change this to your actual image path
image_path = 'C:\\Users\\Admin\\Desktop\\miniprjct_3rd_year\\3rdyear_miniprjct\\images\\obj.jpg'

os.system(f'python detect.py --source {image_path} --weights yolov5s.pt --conf 0.5')

result_image_path = 'runs/detect/exp4/obj.jpg'  # Change to match the actual output path
image = cv2.imread(result_image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()



import io
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from white_box_cartoonizer.cartoonize import WB_Cartoonize

# Disable TensorFlow v2 behavior
tf.disable_v2_behavior()

# Path to the model
model_dir = os.path.abspath("white_box_cartoonizer/saved_models/")

# Initialize Cartoonizer
wb_cartoonizer = WB_Cartoonize(model_dir, gpu=True)

def convert_bytes_to_image(img_bytes):
    """Convert bytes to numpy array."""
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode == "RGBA":
        image = Image.new("RGB", pil_image.size, (255, 255, 255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    return np.array(image)

def apply_gamma_correction(image, gamma=1.2):
    """Apply gamma correction to brighten image."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def cartoonize_image(image_path, output_path):
    # Read the input image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    # Convert BGR to RGB before passing to model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Cartoonize the image
    cartoon_image = wb_cartoonizer.infer(image_rgb)

    # Ensure values are in 0-255 range
    cartoon_image = np.clip(cartoon_image, 0, 255).astype(np.uint8)

    # Apply gamma correction (to reduce darkening effect)
    cartoon_image = apply_gamma_correction(cartoon_image, gamma=1.2)

    # Save the cartoonized image (convert RGB back to BGR)
    cv2.imwrite(output_path, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
    print(f"Cartoonized image saved to {output_path}")

def cartoonize_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB before passing to model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Cartoonize the frame
        cartoon_frame = wb_cartoonizer.infer(frame_rgb)

        # Ensure values are in 0-255 range
        cartoon_frame = np.clip(cartoon_frame, 0, 255).astype(np.uint8)

        # Apply gamma correction
        cartoon_frame = apply_gamma_correction(cartoon_frame, gamma=1.2)

        # Convert RGB back to BGR
        cartoon_bgr = cv2.cvtColor(cartoon_frame, cv2.COLOR_RGB2BGR)

        out.write(cartoon_bgr)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Cartoonized video saved to {output_video_path}")

if __name__ == "__main__":
    choice = input("Do you want to cartoonize an image or video? (Enter 'image' or 'video'): ").strip().lower()

    if choice == 'image':
        input_image_path = input("Enter the full path of the image you want to cartoonize: ").strip()
        output_image_path = input("Enter the output image path (e.g., cartoonized_image.jpg): ").strip()
        cartoonize_image(input_image_path, output_image_path)

    elif choice == 'video':
        input_video_path = input("Enter the full path of the video you want to cartoonize: ").strip()
        output_video_path = input("Enter the output video path (e.g., cartoonized_video.mp4): ").strip()
        cartoonize_video(input_video_path, output_video_path)

    else:
        print("Invalid choice. Please enter 'image' or 'video'.")

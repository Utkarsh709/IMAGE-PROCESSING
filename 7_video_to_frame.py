# break video into multiple images and store in a folder
import cv2
import os

video_path = r"D:\Image_Processing and Computer Vision\pirates1.mp4"
output_folder = r"D:\Image_Processing and Computer Vision\video_to_frame_7"

# create folder if not exists
os.makedirs(output_folder, exist_ok=True)

vidcap = cv2.VideoCapture(video_path)

count = 0

while True:
    ret, image = vidcap.read()
    if not ret:
        break

    frame_path = os.path.join(output_folder, f"img_{count}.jpg")
    cv2.imwrite(frame_path, image)

    cv2.imshow("Frame", image)

    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()
print(f"Total frames saved: {count}")

# access video from youtube
import cv2
import yt_dlp

url = "https://www.youtube.com/watch?v=t0Q2otsqC4I"

ydl_opts = {
    'format': 'best[ext=mp4]',
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    video_url = info['url']

cap = cv2.VideoCapture(video_url)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow("Colorframe", frame)
    cv2.imshow('gray',gray)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

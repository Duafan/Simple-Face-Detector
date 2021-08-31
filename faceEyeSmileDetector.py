import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) #lebar cam
cam.set(4, 480) #tinggi cam

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
smileDetector = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceDetector.detectMultiScale(gray, 1.3, 4) #frame, scaleFactor, minNeighbors
    for (x,y,w,h) in face:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2) #warna Cyan
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeDetector.detectMultiScale(roi_gray, 1.5, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2) #warna hijau

        smile = smileDetector.detectMultiScale(roi_gray, 1.5, 15)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2) #warna merah

    cv2.imshow("Webcam Color", frame)
    cv2.imshow("Webcam Gray", gray)
    
    k =  cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'): #tekan ESC atau q untuk keluar
        break
cam.release()
cv2.destroyAllWindows()

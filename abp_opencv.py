import cv2 

cascpath = "haarcascade_frontalcatface.xml"

plate = cv2.CascadeClassifier(cascpath)

video_capture = cv2.VideoCapture(0)


while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plates = plate.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in plates:
        cv2.rectangle(frame, (x,y), (x+w , y+h), (0,255,0),2 )
    
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord ("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
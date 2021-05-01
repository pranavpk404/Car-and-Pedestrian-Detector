import cv2

car_file = cv2.CascadeClassifier("cars.xml")
Pedestrain_file = cv2.CascadeClassifier("haarcascade_fullbody.xml")
webcam = cv2.VideoCapture("test.mp4")
while True:
    su_frame_read, frame = webcam.read()

    bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_file.detectMultiScale(bw_img)
    people = Pedestrain_file.detectMultiScale(bw_img)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x + 1, y + 1), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Car", (x, y + h + 40), fontScale=1, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 255))

    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Pedestrian", (x, y + h + 40), fontScale=1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    color=(0, 255, 0))

    cv2.imshow("Face Detector", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()

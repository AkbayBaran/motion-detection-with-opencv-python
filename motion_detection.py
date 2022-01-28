import cv2

yakala = cv2.VideoCapture(0)
tanimlama = False

while yakala.isOpened():
    ret, frame1 = yakala.read()
    ret, frame2 = yakala.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if tanimlama:
            cv2.circle(frame1, (570, 100), 15, (0, 255, 0), -1)
        else:
            cv2.circle(frame1, (570, 100), 15, (0, 0, 255), -1)

        if cv2.contourArea(c) < 10000:
            continue

        x, y, w, h, = cv2.boundingRect(c)
        tanimlama = True
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        tanimlama = False

    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow("baran", frame1)

import cv2
import os

cam = cv2.VideoCapture(0)

cv2.namedWindow("Capturing")

img_counter = 0

f = open('cap.csv', 'a')

while True:
    ret, frame = cam.read()
    cv2.imshow("Capturing", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == ord('q'):
        # q pressed
        img_name = "cap{}.png".format(img_counter)
        cv2.imwrite(os.path.join('capture/', img_name), frame)
        print("{} written!, Dab detected".format(img_name))
        f.write(img_name + ',' + 'yes' + '\n')
        img_counter += 1
    elif k%256 == ord('p'):
        # p pressed
        img_name = "cap{}.png".format(img_counter)
        cv2.imwrite(os.path.join('capture/', img_name), frame)
        print("{} written, Dab not detected!".format(img_name))
        f.write(img_name + ',' + 'no' + '\n')
        img_counter += 1

f.close()

cam.release()

cv2.destroyAllWindows()
import cv2
import os

name = "admin"
save_path = f"dataset/{name}"

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
print("Press S to capture image | Q to quit")

while True:
    ret, frame = cap.read()

    cv2.imshow("Admin Face Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        count += 1
        img_path = f"{save_path}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")

    if key == ord('q') or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()

print("Admin dataset created!")
import cv2
import sys

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        sys.exit('❌ Cannot open webcam')
    print('✅ Webcam opened – press q to quit')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('⚠️ Frame capture failed')
            break
        cv2.imshow('Sentinel – Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

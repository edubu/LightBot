import cv2

def main():
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(2)
    
    if not cap0.isOpened():
        print("Could not open cap0")
    if not cap1.isOpened():
        print("Could not open cap1")
    
    caps = [cap0, cap1]
        
    while True:
        
        # read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        cv2.imshow('cam1', frame1)
        cv2.imshow('cam0', frame0)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    for cap in caps:
        cap.release()


if __name__ == "__main__":
    main()
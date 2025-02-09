import cv2

# Open de camera
cap = cv2.VideoCapture(0)  # Gebruik 0 voor de standaard camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Camera", frame)  # Toon de video in een GUI-venster
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Stoppen met 'q'
        break

cap.release()
cv2.destroyAllWindows()
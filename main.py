import cv2
import numpy as np
import pyautogui

# Initialisation et configuration de la capture vidéo
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables globales
points = []  # Pour stocker les points sélectionnés par l'utilisateur
projector_width, projector_height = 1920, 1080  # Résolution du projecteur

# Fonction pour sélectionner les coins de l'écran/projecteur
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 4:
            cv2.destroyAllWindows()

# Fonction pour détecter le point laser et déplacer la souris
def detect_laser_and_move_mouse(frame, projector_width, projector_height):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_frame, 250, 255, cv2.THRESH_BINARY)  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 5:  
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                screen_x, screen_y = pyautogui.size()
                mouse_x = np.interp(cX, [0, projector_width], [0, screen_x])
                mouse_y = np.interp(cY, [0, projector_height], [0, screen_y])

                pyautogui.moveTo(mouse_x, mouse_y)

# Boucle principale
cv2.namedWindow("Select Corners")
cv2.setMouseCallback("Select Corners", select_points)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if len(points) == 4:
        src_pts = np.array(points, dtype="float32")
        dst_pts = np.array([[0, 0], [projector_width - 1, 0], [0, projector_height - 1], [projector_width - 1, projector_height - 1]], dtype="float32")
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_frame = cv2.warpPerspective(frame, matrix, (projector_width, projector_height))
        
        detect_laser_and_move_mouse(warped_frame, projector_width, projector_height)  
        
        cv2.imshow("Warped Screen", warped_frame)  
    else:
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
        cv2.imshow("Select Corners", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
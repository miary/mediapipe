import cv2
import mediapipe
 
drawingModule = mediapipe.solutions.drawing_utils
faceModule = mediapipe.solutions.face_mesh
 
circleDrawingSpec = drawingModule.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
lineDrawingSpec = drawingModule.DrawingSpec(thickness=1, color=(0,255,0))
 
with faceModule.FaceMesh(static_image_mode=True) as face:
    image = cv2.imread("sleepasian.mov")
 
    results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
 
    if results.multi_face_landmarks != None:
        for faceLandmarks in results.multi_face_landmarks:
            drawingModule.draw_landmarks(image, faceLandmarks, faceModule.FACE_CONNECTIONS, circleDrawingSpec, lineDrawingSpec)
 
    cv2.imshow('Test image', image)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
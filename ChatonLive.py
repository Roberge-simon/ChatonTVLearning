import cv2
import numpy as np
import ChatonTV

def ImageToX(image):
    img = cv2.resize(image,(128, 128), interpolation = cv2.INTER_CUBIC)
    flat = np.reshape(img, np.prod(img.shape))
    return np.array([flat]).T
    
def Run():
    cap = cv2.VideoCapture(0)

    name = "PasChaton/VideA"

    i = 0
    while(True):
        
        ret, frame = cap.read()
        
        X = ImageToX(frame)
        chaton = ChatonTV.Predict(X, ChatonTV.w, ChatonTV.b)[0]
        
        # Display the resulting frame
        
        if chaton:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Chaton!',(10,128), font, 4,(255,255,255),2,cv2.LINE_AA)
        
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
Run()
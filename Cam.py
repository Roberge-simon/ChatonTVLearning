import numpy as np
import cv2

cap = cv2.VideoCapture(0)

name = "Simon/Yes"

i = 0
while(True):
	
    ret, frame = cap.read()
    i+=1	

    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    #if i %5 == 0 :
    cv2.imwrite("C:/Users/Rober/Desktop/Program/Images/"+name+ str(i)+".jpg", frame )
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
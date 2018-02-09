import cv2
import os

def ApplyFilter(input_folder, output_folder, prefix, filterFunc):
    i =0
    for filename in os.listdir(input_folder):
        in_name = input_folder + '/' + filename
        out_name = output_folder + '/' + prefix + str(i) + '.png'
        print ("processing " + in_name)
        
        img = cv2.imread(in_name, cv2.IMREAD_COLOR)
        filtered = filterFunc(img)
        cv2.imwrite(out_name, filtered)
        
        i = i+1
        

def Shrink(width, height):
    def closure(image):
        return cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
    return closure

no_path = "C:/Users/Rober/Desktop/Program/Images/PasChaton"
yes_path ="C:/Users/Rober/Desktop/Program/Images/Chatons"
out_path ="C:/Users/Rober/Desktop/Program/Images/Processed"
 
ApplyFilter(no_path, out_path, "no", Shrink(128,128))
ApplyFilter(yes_path, out_path, "yes", Shrink(128,128))
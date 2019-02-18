from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image2.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"),minimum_percentage_probability=10)
c1,c2=0,0
#print(detections)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    if(eachObject["name"]=="person"):
        c1=c1+1
    if(eachObject["name"]=="cars" or eachObject["name"]=="bus" or eachObject["name"]=="truck" or eachObject["name"]=="bicycle"):
        c2=c2+1
print("No. of persons ",c1)
 
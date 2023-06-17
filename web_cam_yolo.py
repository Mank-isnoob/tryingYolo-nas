from super_gradients.training import models

yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco") # imports our pre-trained model

url = "cars.jpg" # you can use your own file for image for detection
yolo_nas_l.predict(url, conf=0.25).show()   #this can predict all classes from coco with yolo_nas_l

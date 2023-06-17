from super_gradients.training import models

yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

url = "cars.jpg"
yolo_nas_l.predict(url, conf=0.25).show()   #this can predict all models with yolo_nas_l
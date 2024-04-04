import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

## load model
model  = tf.keras.models.load_model('entrainement_detection_maladie.keras') 
model.summary()
## visuaalizing single image of test set
import cv2 as cv 
image_path = "C:\\Users\\hiche\\Desktop\\PFE EX\\Test 1\\healthy.png"
#reading image
img = cv.imread(image_path)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB) 
plt.imshow(img)
plt.title("test image")
plt.xticks([])#tfassekh l'axe des abscisses 
plt.yticks([])#tfassekh l'axe des ordonnées
plt.show()
## test model
image =tf.keras.preprocessing.image.load_img(image_path,target_size = (128 ,128 ))
input_arr = tf.keras.preprocessing.image.img_to_array(image)#neurone te9bel kan array form hadheka alech 7awelnaha
input_arr =np.array([input_arr])## conversion single image to batch yaani mdakhlyn bacth wa7da
print (input_arr.shape)
#perform model prediction 
prediction = model.predict(input_arr) 
print (prediction , prediction.shape)#
result_index = np.argmax(prediction) ## traje3lik el classe ely akber e7timalia enno s7i7
print(result_index)
class_name  = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 
                      'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'TomatoTomato_Yellow_Leaf_Curl_Virus', 
                      'Tomato_mosaic_virus', 'healthy', 'powdery_mildew']
#Display result prediction 
# Assuming class_name is a set
# Convert it to a list
class_name = list(class_name)
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name : {model_prediction}")
plt.xticks([])#tfassekh l'axe des abscisses 
plt.yticks([])#tfassekh l'axe des ordonnées
plt.show()
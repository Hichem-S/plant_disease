#importing libraries
import tensorflow as tf 
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import json
from tensorflow.keras.models import load_model
import seaborn as sns 
import pandas as pd
data_dir = "train 1"
data_dirr = "valid 1"
#Data preprocessing
#training image processing 
training_set= tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",#i said to tensorflow go inside this folder (dataset) and go inside my train folder whatever the name of directory select it as my label 
    label_mode="categorical",#ki nabdo net3amlo m3a akther men 2 classes nesta3mlo categorical w ta3ni les labels codées as categorical vector
    color_mode="rgb",
    batch_size=32,#ki nebo nsar3o l'entrainement nzido feha n7otoha 64 ou 128
    image_size=(128, 128),
    shuffle=True,# at the time of feeding to my model for the training shuffle the entire thing and pass it and it reduce the biasness of the model if i shuffle it my model will learn from all end if i don't shuffle it it will pass some classes
)
# validation images preprocessing 
validation_set= tf.keras.utils.image_dataset_from_directory(
    data_dirr,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
)
for x,y in training_set: #x value of each pixel y label
    print(x,x.shape)
    print(y,y.shape)
    break
# Building model 
"""
model = Sequential()
#Building convolutions layers
#convolution process
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128,128,3], kernel_regularizer=tf.keras.regularizers.l2(0.001)))#nesta3mlo 32 filters tkhrajellna 32 different matrix(feature map) kernel size yaany input lowel matrice (3,3)
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu' ,  kernel_regularizer=tf.keras.regularizers.l2(0.001)))#nesta3mlo 32 filters tkhrajellna 32 different matrix(feature map) kernel size yaany input lowel matrice (3,3)
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu' , kernel_regularizer=tf.keras.regularizers.l2(0.001)))#nesta3mlo 32 filters tkhrajellna 32 different matrix(feature map) kernel size yaany input lowel matrice (3,3)
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu' , kernel_regularizer=tf.keras.regularizers.l2(0.001)))#nesta3mlo 32 filters tkhrajellna 32 different matrix(feature map) kernel size yaany input lowel matrice (3,3)
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu' , kernel_regularizer=tf.keras.regularizers.l2(0.001)))#padding means whatever the input are coming take the same size
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Dropout(0.25)) #to avoid overfiting
#create mcuh of feature map to understand the caracteristiques and proprities of the image
model.add(Flatten())
#Add fully connected layer
model.add(Dense(units=1500,activation='relu'))#units means how many number of neurones i want
model.add(Dropout(0.5))
#output layer  
model.add(Dense(units=11,activation='softmax'))#10 neurones in output 3la gued les classes ely 3endy softmax give me probality of each class or neurones
#compiling the model
# Compile the model with appropriate loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00001) ,loss='categorical_crossentropy',metrics=['accuracy']) #Using adam as optimizer and categorical cross we use optimizer to minimize the loss 
model.summary()
#Training model
# Assuming your training and validation datasets are properly created
# training_set and validation_set should yield tuples (x_train, y_train) and (x_val, y_val) respectively
training_history = model.fit(training_set, epochs=25, validation_data=validation_set)
#Saving Model 
model.save('entrainement_detection_maladie.keras')
print("Model saved")"""
#Loading Model


# Spécifiez le chemin vers le fichier où votre modèle est enregistré
model_path = "entrainement_detection_maladie.keras"

# Chargez le modèle à partir du fichier
loaded_model = load_model(model_path)
"""
# Évaluer le modèle sur l'ensemble d'entraînement
training_loss, training_accuracy = loaded_model.evaluate(training_set)
print('Loss on training set:', training_loss)
print('Accuracy on training set:', training_accuracy)

# Évaluer le modèle sur l'ensemble de validation
validation_loss, validation_accuracy = loaded_model.evaluate(validation_set)
print('Loss on validation set:', validation_loss)
print('Accuracy on validation set:', validation_accuracy)"""

"""
# Créer un dictionnaire contenant les données de l'historique d'entraînement
training_history = {
    "loss": [2.5943, 1.9331, 1.6753, 1.5283, 1.4137, 1.3205, 1.2542, 1.1822, 1.1206, 1.0759, 1.0272, 0.9874, 0.9462, 0.9077, 0.8848, 0.8478, 0.8185, 0.7910, 0.7589, 0.7381, 0.7163, 0.6974, 0.6805, 0.6576, 0.6471],
    "accuracy": [0.3165, 0.5496, 0.6252, 0.6637, 0.6982, 0.7278, 0.7474, 0.7667, 0.7861, 0.8025, 0.8174, 0.8305, 0.8431, 0.8554, 0.8620, 0.8755, 0.8834, 0.8925, 0.9041, 0.9108, 0.9181, 0.9228, 0.9275, 0.9365, 0.9389],
    "val_loss": [1.9700, 1.6356, 1.4740, 1.3498, 1.2803, 1.2049, 1.1463, 1.0880, 1.0667, 1.0345, 1.0053, 0.9719, 0.9457, 0.9260, 0.9217, 0.8952, 0.8982, 0.8786, 0.8734, 0.8865, 0.8906, 0.8548, 0.8453, 0.8353, 0.8447],
    "val_accuracy": [0.5527, 0.6469, 0.6808, 0.7232, 0.7382, 0.7670, 0.7831, 0.8039, 0.8050, 0.8168, 0.8292, 0.8375, 0.8499, 0.8514, 0.8540, 0.8640, 0.8611, 0.8706, 0.8704, 0.8680, 0.8706, 0.8827, 0.8850, 0.8851, 0.8885]
}

# Spécifiez le chemin du fichier JSON où vous souhaitez enregistrer l'historique
history_file_path = "training_history.json"

# Enregistrez l'historique d'entraînement dans un fichier JSON
with open(history_file_path, 'w') as json_file:
    json.dump(training_history, json_file)

print("L'historique d'entraînement a été enregistré dans le fichier :", history_file_path)

"""
# Charger les données du fichier JSON
with open('training_history.json', 'r') as file:
    history = json.load(file)

# Extraire les valeurs d'accuracy de l'historique d'entraînement
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']

# Créer une liste d'épochs pour l'axe x
epochs = range(1, len(accuracy) + 1)

# Tracer le graphique
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Afficher le graphique
plt.show()

## some other metrics for model evaluation 
class_name = validation_set.class_names
print (class_name)
test_set = tf.keras.utils.image_dataset_from_directory(
    data_dirr,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
)
# Faites des prédictions sur l'ensemble de données de test
y_predictions = loaded_model.predict(test_set)

# Affichez les prédictions
print(y_predictions)

predicted_categories = tf.argmax(y_predictions, axis=1) # traje3 maximun value ely predictions bech tkharjo axis = 1 ya3ny bech yemchy b façon sequential w traje3 index te3o 
print(predicted_categories)
true_categories = tf.concat([y for x,y in test_set] , axis= 0)
print (true_categories)
y_true = tf.argmax(true_categories, axis=1)
print(y_true)
#Precision , Recall = f1 score 
from sklearn.metrics import classification_report , confusion_matrix
print (classification_report (y_true,predicted_categories,target_names = class_name))
cm = confusion_matrix (y_true , predicted_categories)
print (cm) 

#confusion matrix visualization 
plt.figure(figsize = [40 , 40])
sns.heatmap(cm, annot=True, annot_kws={'size': 20})
plt.xlabel("predicted class" , fontsize = 20)
plt.ylabel("Actual class" , fontsize = 20)
plt.title ("Plant Disease Prediction Confusion Matrix " , fontsize = 30)
plt.show()


import streamlit as st 
import tensorflow as tf
import numpy as np
import time

# Tensorflow model prediction
def predict_model(test_image):
    # Ajouter une légère pause pour simuler le temps de traitement
    time.sleep(2)
    model = tf.keras.models.load_model('entrainement_detection_maladie.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)  
    result_index = np.argmax(prediction)
    return result_index

# Définition des noms de classe
class_name  = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 
                      'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'TomatoTomato_Yellow_Leaf_Curl_Virus', 
                      'Tomato_mosaic_virus', 'healthy', 'powdery_mildew']

# Barre latérale
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page" , ["Home","About" , "Disease Recognition"])

# Page d'accueil
if app_mode == 'Home':
    st.header("Plant Disease Recognition System")
    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Plant Disease Recognition System!
        Our mission is to help in identifying plant disease efficiently. 
        Upload an image of a tomato plant, and our system will analyze it to detect any signs of diseases. 
        Together, let's protect our crops and ensure a healthier harvest!
        ### How it works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.
        ### Why Choose Us
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
        ### Get started 
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System
        ### About Us 
        Learn more about the project, our team, and our goals on the **About** Page  
    """)
# Page À propos
elif app_mode == 'About':
    st.subheader("About")
    st.markdown("""
        ### About Dataset
        This data is recreated using offline augmentation from the original dataset. 
        The original dataset can be found on this GitHub repo. 
        This dataset consists of about 25k regular images of healthy and diseased crop leaves which is categorized into 11 different classes.
        ### Content 
        1. Train (25,984 images)
        2. Valid (6,698 images)
        3. Test (11 images)
    """)
# Page de prédiction
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image:")
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        # Bouton de prédiction
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                result_index = predict_model(test_image)
                st.success("Model is predicting it's a {}".format(class_name[result_index]))

from tensorflow.keras.models import load_model
import numpy as np

# Load model once at import
model = load_model('digit_classifier.h5')

def predict_digit_from_array(pixel_array):
    pixel_array = pixel_array.reshape(1, 784)
    pixel_array = pixel_array.astype('float32') / 255.0
    prediction = model.predict(pixel_array)
    return int(np.argmax(prediction))
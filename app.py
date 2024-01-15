from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model (assuming it's in the same directory)
model_path = 'D:/Model backend/model1.keras'
class_names = ['Bacterial Leaf Blight', 'Blast', 'Brownspot', 'Healthy']
model = None

# Ensure that the model is loaded only once when the server starts
def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)

# Prediction function
def predict_disease(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])
    predicted_label = class_names[predicted_class]
    return predicted_label, confidence, predictions[0].tolist()

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Save the file temporarily
            temp_path = 'temp_image.jpg'
            file.save(temp_path)

            # Make prediction
            predicted_label, confidence, probabilities = predict_disease(temp_path)
            os.remove(temp_path)  # Remove the temporary file

            # Return the prediction result with additional details
            return jsonify({
                'predicted_class': predicted_label,
                'confidence': float(confidence),
                'probabilities': probabilities
            })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model()  # Load the model when the server starts
    app.run(debug=True)

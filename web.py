from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
import os
from gtts import gTTS
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
model = tf.keras.models.load_model('traffic_sign_vgg16.h5')

# Move class_labels to a more global scope
label = {0:"Speed Limit 5", 1:"Speed Limit 15", 2:"Speed Limit 30",
         3:"Speed Limit 40", 4:"Speed Limit 50", 5:"Speed Limit 60",
         6:"Speed Limit 70", 7:"Speed Limit 80", 8:"Don't go straight or left",
         9:"Don't go straight or right", 10:"Don't go straight", 11:"No Left",
         12:"Don't go right or left", 13:"Don't go right", 14:"No Overtake from Left",
         15:"No U-turn", 16:"No Cars", 17:"No Horn", 18:"Speed Limit (40km/h)",
         19:"Speed Limit (50km/h)", 20:"Go straight or right", 21:"Watch out for cars",
         22:"Go left", 23:"Go left or right", 24:"Go right", 25:"Keep Left",
         26:"Keep Right", 27:"Roundabout mandatory", 28:"Go Straight",
         29:"Horn", 30:"Bicycle Crossing", 31:"U-turn", 32:"Road Divider",
         33:"Traffic Signals", 34:"Danger ahead", 35:"Zebra Crossing",
         36:"Bicycle Crossing", 37:"Children Crossing", 38:"Dangerous curve to the left",
         39:"Dangerous curve to the right", 40:"Unknown 1", 41:"Unknown 2", 42:"Unknown 3",
         43:"Go right or straight", 44:"Go left or straight", 45:"Unknown 4",
         46:"Zigzag curve", 47:"Train Crossing", 48:"Under construction", 49:"Unknown 5",
         50:"Fences", 51:"Heavy Vehicle Accidents", 52:"Unknown 6", 53:"Give way",
         54:"No Stopping", 55:"No Entry", 56:"Yield", 57:"Unknown 8"}

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_prediction(predicted_class_label):
    # Default message if the predicted_class is not recognized
    message = "Error: Unable to load and preprocess the test image."

    if predicted_class_label == 0:
        message = "Alert: Speed Limit 5. Drive with caution."
    elif predicted_class_label == 1:
        message = "Alert: Speed Limit 15. Drive with caution."
    elif predicted_class_label == 2:
        message = "Alert: Speed Limit 30. Drive with caution."
    elif predicted_class_label == 3:
        message = "Alert: Speed Limit 40. Drive with caution."
    elif predicted_class_label == 4:
        message = "Alert: Speed Limit 50. Drive with caution."
    elif predicted_class_label == 5:
        message = "Alert: Speed Limit 60. Drive with caution."
    elif predicted_class_label == 6:
        message = "Alert: Speed Limit 70. Drive with caution."
    elif predicted_class_label == 7:
        message = "Alert: Speed Limit 80. Drive with caution."
    elif predicted_class_label == 8:
        message = "Alert: Don't go straight or left. Drive with caution."
    elif predicted_class_label == 9:
        message = "Alert: Don't go straight or right. Drive with caution."
    elif predicted_class_label == 10:
        message = "Alert: Don't go straight. Drive with caution."
    elif predicted_class_label == 11:
        message = "Alert: No Left. Drive with caution."
    elif predicted_class_label == 12:
        message = "Alert: Don't go right or left. Drive with caution."
    elif predicted_class_label == 13:
        message = "Alert: Don't go right. Drive with caution."
    elif predicted_class_label == 14:
        message = "Alert: No Overtake from Left. Drive with caution."
    elif predicted_class_label == 15:
        message = "Alert: No U-turn. Drive with caution."
    elif predicted_class_label == 16:
        message = "Alert: No Cars. Drive with caution."
    elif predicted_class_label == 17:
        message = "Alert: No Horn. Drive with caution."
    elif predicted_class_label == 18:
        message = "Alert: Speed Limit (40km/h). Drive with caution."
    elif predicted_class_label == 19:
        message = "Alert: Speed Limit (50km/h). Drive with caution."
    elif predicted_class_label == 20:
        message = "Alert: Go straight or right. Drive with caution."
    elif predicted_class_label == 21:
        message = "Alert: Watch out for cars. Drive with caution."
    elif predicted_class_label == 22:
        message = "Alert: Go left. Drive with caution."
    elif predicted_class_label == 23:
        message = "Alert: Go left or right. Drive with caution."
    elif predicted_class_label == 24:
        message = "Alert: Go right. Drive with caution."
    elif predicted_class_label == 25:
        message = "Alert: Keep Left. Drive with caution."
    elif predicted_class_label == 26:
        message = "Alert: Keep Right. Drive with caution."
    elif predicted_class_label == 27:
        message = "Alert: Roundabout mandatory. Drive with caution."
    elif predicted_class_label == 28:
        message = "Alert: Go Straight. Drive with caution."
    elif predicted_class_label == 29:
        message = "Alert: Horn. Drive with caution."
    elif predicted_class_label == 30:
        message = "Alert: Bicycle Crossing. Drive with caution."
    elif predicted_class_label == 31:
        message = "Alert: U-turn. Drive with caution."
    elif predicted_class_label == 32:
        message = "Alert: Road Divider. Drive with caution."
    elif predicted_class_label == 33:
        message = "Alert: Traffic Signals. Drive with caution."
    elif predicted_class_label == 34:
        message = "Alert: Danger ahead. Drive with caution."
    elif predicted_class_label == 35:
        message = "Alert: Zebra Crossing. Drive with caution."
    elif predicted_class_label == 36:
        message = "Alert: Bicycle Crossing. Drive with caution."
    elif predicted_class_label == 37:
        message = "Alert: Children Crossing. Drive with caution."
    elif predicted_class_label == 38:
        message = "Alert: Dangerous curve to the left. Drive with caution."
    elif predicted_class_label == 39:
        message = "Alert: Dangerous curve to the right. Drive with caution."
    elif predicted_class_label == 40:
        message = "Alert: Unknown 1. Drive with caution."
    elif predicted_class_label == 41:
        message = "Alert: Unknown 2. Drive with caution."
    elif predicted_class_label == 42:
        message = "Alert: Unknown 3. Drive with caution."
    elif predicted_class_label == 43:
        message = "Alert: Go right or straight. Drive with caution."
    elif predicted_class_label == 44:
        message = "Alert: Go left or straight. Drive with caution."
    elif predicted_class_label == 45:
        message = "Alert: Unknown 4. Drive with caution."
    elif predicted_class_label == 46:
        message = "Alert: Zigzag curve. Drive with caution."
    elif predicted_class_label == 47:
        message = "Alert: Train Crossing. Drive with caution."
    elif predicted_class_label == 48:
        message = "Alert: Under construction. Drive with caution."
    elif predicted_class_label == 49:
        message = "Alert: Unknown 5. Drive with caution."
    elif predicted_class_label == 50:
        message = "Alert: Fences. Drive with caution."
    elif predicted_class_label == 51:
        message = "Alert: Heavy Vehicle Accidents. Drive with caution."
    elif predicted_class_label == 52:
        message = "Alert: Unknown 6. Drive with caution."
    elif predicted_class_label == 53:
        message = "Alert: Give way. Drive with caution."
    elif predicted_class_label == 54:
        message = "Alert: No Stopping. Drive with caution."
    elif predicted_class_label == 55:
        message = "Alert: No Entry. Drive with caution."
    elif predicted_class_label == 56:
        message = "Alert: Yield. Drive with caution."
    elif predicted_class_label == 57:
        message = "Alert: Unknown 8. Drive with caution."


    # Render HTML template with the prediction text
    render_result = render_template('result.html', prediction_text=message)

    # Convert the message to speech
    text_to_speech(message)

    return render_result

def text_to_speech(message):
    # Use gTTS to convert text to speech
    tts = gTTS(text=message, lang='en')
    
    # Save the speech as an audio file
    audio_file_path = 'static/audio/output.mp3'
    tts.save(audio_file_path)

    # Play the audio file (you can customize this based on your needs)
    os.system(f'start {audio_file_path}')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return render_template('result.html', prediction_text="Error: No file part")

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return render_template('result.html', prediction_text="Error: No selected file")

    # Check if the file is allowed
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        # Save the file to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Function to preprocess a test image
        def preprocess_test_image(unknown_image_path):
            unknown_image = image.load_img(unknown_image_path, target_size=(224, 224))
            unknown_image_array = image.img_to_array(unknown_image)
            unknown_image_array = unknown_image_array / 255.0  # Normalize the pixel values
            return unknown_image_array

        # Preprocess the test image
        test_image_array = preprocess_test_image(file_path)

        # Check if the image is not None
        if test_image_array is not None:
            # Expand the dimensions to match the input shape of the model
            test_image_array = np.expand_dims(test_image_array, axis=0)
            # Make predictions
            predictions = model.predict(test_image_array)

            # Decode predictions
            predicted_class_index = np.argmax(predictions)
            class_indices={'0': 0,
                 '1': 1,
                 '10': 2,
                 '11': 3,
                 '12': 4,
                 '13': 5,
                 '14': 6,
                 '15': 7,
                 '16': 8,
                 '17': 9,
                 '18': 10,
                 '19': 11,
                 '2': 12,
                 '20': 13,
                 '21': 14,
                 '22': 15,
                 '23': 16,
                 '24': 17,
                 '25': 18,
                 '26': 19,
                 '27': 20,
                 '28': 21,
                 '29': 22,
                 '3': 23,
                 '30': 24,
                 '31': 25,
                 '32': 26,
                 '33': 27,
                 '34': 28,
                 '35': 29,
                 '36': 30,
                 '37': 31,
                 '38': 32,
                 '39': 33,
                 '4': 34,
                 '40': 35,
                 '41': 36,
                 '42': 37,
                 '43': 38,
                 '44': 39,
                 '45': 40,
                 '46': 41,
                 '47': 42,
                 '48': 43,
                 '49': 44,
                 '5': 45,
                 '50': 46,
                 '51': 47,
                 '52': 48,
                 '53': 49,
                 '54': 50,
                 '55': 51,
                 '56': 52,
                 '57': 53,
                 '6': 54,
                 '7': 55,
                 '8': 56,
                 '9': 57}
            predicted_class_label = [k for k, v in class_indices.items() if v == predicted_class_index][0]
            predicted_class = int(predicted_class_label)
            #predicted_class_name = label[predicted_class]
            
            # Call the process_prediction function
            result = process_prediction(predicted_class)
            return result

            #return render_template('result.html', prediction_text=f"Predicted class: {predicted_class_name}")

    else:
        return render_template('result.html', prediction_text="Error: Unsupported file format")

if __name__ == '__main__':
    app.run(port=8000)

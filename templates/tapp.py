from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import pyodbc
import base64
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from sklearn.cluster import KMeans

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Ensure to set a secret key for session management

# Configure your database connection here
server = 'DESKTOP-16A7BAU'
database = 'fashion'
username = 'cube_sl'
password = '123'
driver = '{ODBC Driver 17 for SQL Server}'

connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Load the trained model
model = load_model("skin_tone_classifier.h5")

categories = ['Black', 'Brown', 'Dark-brown', 'Olive', 'White']
img_size = 128  # This should match the size used during training

@app.route('/')
def welcome():
    return render_template('capture_video.html')

# Other routes...

@app.route('/submit_snapshots', methods=['POST'])
def submit_snapshots():
    data = request.get_json()
    snapshots = data.get('snapshots')
    phone_number = session.get('phone_number')

    if not phone_number:
        return jsonify({'message': 'Failed to retrieve phone number for user'})

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # Store snapshots in the database
        for snapshot in snapshots:
            img_data = base64.b64decode(snapshot.split(',')[1])
            cursor.execute("INSERT INTO video (PhoneNumber, Snapshot) VALUES (?, ?)", phone_number, img_data)

        conn.commit()

        # Predict skin tone category based on the last snapshot
        last_snapshot = base64.b64decode(snapshots[-1].split(',')[1])
        img_array = np.asarray(bytearray(last_snapshot), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Resize and normalize image
        img_resized = cv2.resize(img, (img_size, img_size))
        img_normalized = img_resized / 255.0
        img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 3))

        # Perform prediction
        prediction = model.predict(img_reshaped)
        predicted_class = np.argmax(prediction)
        predicted_tone = categories[predicted_class]

        # Save the predicted skin tone in the session
        session['predicted_tone'] = predicted_tone

        cursor.close()
        conn.close()

        return jsonify({'message': 'Snapshots saved and skin tone predicted successfully'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'message': 'Failed to save snapshots and predict skin tone'})

def classify_skin_tone_from_base64(base64_image):
    img_data = base64.b64decode(base64_image.split(',')[1])
    img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 3))
    prediction = model.predict(img_reshaped)
    predicted_class = np.argmax(prediction)
    return categories[predicted_class]


@app.route('/get_skin_tone')
def get_skin_tone():
    predicted_tone = session.get('predicted_tone')

    if not predicted_tone:
        return jsonify({'message': 'No skin tone prediction available'}), 400

    recommended_colors = recommend_colors_for_skin_tone(predicted_tone)

    return jsonify({'predicted_tone': predicted_tone, 'recommended_colors': recommended_colors})

def recommend_colors_for_skin_tone(predicted_tone):
    color_palettes = {
        "Black": ["maroon", "navy", "emerald"],
        "Brown": ["pastels", "soft blues", "lavender"],
        "Dark-brown": ["olive green", "burnt orange", "mustard yellow"],
        "Olive": ["peach", "light turquoise", "coral"],
        "White": ["pastels", "soft blues", "lavender", "burnt orange"]
    }


@app.route('/outfit', methods=['GET'])
def outfit():
    predicted_tone = session.get('predicted_tone', 'Unknown')
    return render_template('outfit.html', predicted_tone=predicted_tone)

if __name__ == '__main__':
    app.run(debug=True)

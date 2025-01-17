from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import pyodbc
import base64
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import uuid
import random


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Ensure to set a secret key for session management

# Configure your database connection here
server = 'DESKTOP-16A7BAU'
database = 'fashion'
username = 'cube_sl'
password = '123'
driver = '{ODBC Driver 17 for SQL Server}'

connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/user_login')
def user_login():
    user_id = session.get('user_id')
    return render_template('user_login.html', user_id=user_id)


@app.route('/admin_login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Admins WHERE username = ? AND password = ?', (username, password))
        account = cursor.fetchone()
        conn.close()

        if account:
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[1]
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Incorrect username/password!', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    flash('You have successfully logged out.', 'success')
    return redirect(url_for('welcome'))


def generate_unique_user_id():
    while True:
        user_id = str(random.randint(100000, 999999))
        with pyodbc.connect(connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user2 WHERE user_id = ?", (user_id,))
            count = cursor.fetchone()[0]
            if count == 0:
                return user_id


@app.route('/submit', methods=['POST'])
def submitInput():
    data = request.json
    phone_number = data.get('phone_number')

    if not phone_number or len(phone_number) != 10:
        return jsonify({'error': 'Invalid phone number'}), 400

    user_id = generate_unique_user_id()  # Generate a unique user_id

    try:
        with pyodbc.connect(connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO user2 (PhoneNumber, user_id) VALUES (?, ?)", (phone_number, user_id))
            conn.commit()

        session['phone_number'] = phone_number
        session['user_id'] = user_id  # Store user_id in session
        return jsonify({'message': 'Registration successful', 'user_id': user_id}), 200
    except Exception as e:
        print(f"Error occurred during registration: {str(e)}")  # Add error logging
        return jsonify({'error': str(e)}), 500


@app.route('/gender', methods=['GET', 'POST'])
def gender():
    if request.method == 'POST':
        data = request.json
        gender = data.get('gender')
        phone_number = data.get('phone_number')

        if not gender:
            return jsonify({'error': 'Gender not provided'}), 400
        if not phone_number or len(phone_number) != 10:
            return jsonify({'error': 'Invalid phone number'}), 400

        try:
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()

            cursor.execute("UPDATE user2 SET gender = ? WHERE PhoneNumber = ?", (gender, phone_number))

            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({'message': 'Gender updated successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return render_template('gender.html')

@app.route('/submit_phone', methods=['POST'])
def submit_phone_number():
    data = request.json
    phone_number = data.get('phone_number')
    gender = data.get('gender')

    if not phone_number or len(phone_number) != 10:
        return jsonify({'error': 'Invalid phone number'}), 400

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user2 WHERE PhoneNumber = ? AND Gender = ?", (phone_number, gender))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            return jsonify({'message': 'Login successful'}), 200
        else:
            return jsonify({'error': 'Invalid phone number or gender'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/material', methods=['GET'])
def material():
    return render_template('material.html')

@app.route('/submit_material', methods=['POST'])
def submit_material():
    data = request.json
    material = data.get('material')
    phone_number = session.get('phone_number')

    if not material:
        return jsonify({'error': 'Material not provided'}), 400

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("UPDATE user2 SET material = ? WHERE PhoneNumber = ?", (material, phone_number))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'Material updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outfitCategory', methods=['GET'])
def outfit_category():
    return render_template('outfitCategory.html')

# Load the trained model
model = load_model("skin_tone_classifier.h5")

categories = ['Black', 'Brown', 'Dark-brown', 'Olive', 'White']
img_size = 128  # This should match the size used during training


@app.route('/capture_video', methods=['GET'])
def capture_video():
    return render_template('capture_video.html')

@app.route('/submit_snapshots', methods=['POST'])
def submit_snapshots():
    data = request.get_json()
    snapshots = data.get('snapshots')
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({'message': 'Failed to retrieve phone number for user'})

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        for snapshot in snapshots:
            img_data = base64.b64decode(snapshot.split(',')[1])
            cursor.execute("INSERT INTO video (user_id, Snapshot) VALUES (?, ?)", user_id, img_data)

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

        return jsonify({'message': 'Snapshots saved successfully'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'message': 'Failed to save snapshots'})

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

@app.route('/outfit')
def outfit():
    try:
        predicted_tone = session.get('predicted_tone')

        if not predicted_tone:
            return "Predicted skin tone not found. Please capture and submit video first."

        recommended_colors = recommend_colors_for_skin_tone(predicted_tone)

        return render_template('outfit.html', predicted_tone=predicted_tone, recommended_colors=recommended_colors)

    except Exception as e:
        print(f"Error in /outfit route: {str(e)}")
        return render_template('error.html', error_message=str(e)), 500

def recommend_colors_for_skin_tone(predicted_tone):
    color_palettes = {
        "Black": ["Red", "Yellow", "Orange"],
        "Brown": ["Green", "Blue", "Purple"],
        "Dark-brown": ["Pink", "Gray", "Black"],
        "Olive": ["Brown", "Gold", "Tan"],
        "White": ["White", "Silver", "Light Blue"]
    }

    return color_palettes.get(predicted_tone, ["Neutral"])


@app.route('/submit', methods=['POST'])
def submit():
    # Add your logic here for handling the POST request to /submit
    return jsonify({'message': 'Submit endpoint reached successfully'}), 200

@app.route('/feedback', methods=['GET', 'POST'])
def submitFeedback():
    if request.method == 'POST':
        try:
            data = request.json
            color = data.get('color')
            size = data.get('size')
            outfit = data.get('outfit')
            overall_appearance = data.get('overall_appearance')

            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()

            cursor.execute("INSERT INTO feedback (color, size, outfit, overall_appearance) VALUES (?, ?, ?, ?)",
                           (color, size, outfit, overall_appearance))

            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({'message': 'Feedback submitted successfully'}), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('feedback.html')

@app.route('/thank')
def thank():
    return render_template('thank.html')

@app.route('/admin_dashboard')
def index():
    if 'loggedin' in session:
        return render_template('admin_dashboard.html')
    else:
        return redirect(url_for('login'))

@app.route('/manage_material', methods=['GET', 'POST'])
def manage_material():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    if request.method == 'POST':
        if 'add' in request.form:
            name = request.form['name']
            description = request.form['description']
            cursor.execute('INSERT INTO MaterialType (name, description) VALUES (?, ?)', (name, description))
            conn.commit()
            flash('Material added successfully!', 'success')
        elif 'drop' in request.form:
            material_id = request.form['material_id']
            cursor.execute('DELETE FROM MaterialType WHERE id = ?', (material_id,))
            conn.commit()
            flash('Material dropped successfully!', 'success')
        return redirect(url_for('manage_material'))
    cursor.execute('SELECT * FROM MaterialType')
    materials = cursor.fetchall()
    conn.close()
    return render_template('manage_material.html', materials=materials)


@app.route('/manage_inventory', methods=['GET', 'POST'])
def manage_inventory():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    if request.method == 'POST':
        if 'add' in request.form:
            material_id = request.form['material_id']
            quantity = request.form['quantity']
            cursor.execute('INSERT INTO Inventory (material_id, quantity) VALUES (?, ?)', (material_id, quantity))
            conn.commit()
            flash('Inventory updated successfully!', 'success')
        elif 'drop' in request.form:
            inventory_id = request.form['inventory_id']
            cursor.execute('DELETE FROM Inventory WHERE id = ?', (inventory_id,))
            conn.commit()
            flash('Inventory item dropped successfully!', 'success')
        return redirect(url_for('manage_inventory'))
    cursor.execute(
        'SELECT Inventory.id, MaterialType.name, Inventory.quantity FROM Inventory JOIN MaterialType ON Inventory.material_id = MaterialType.id')
    inventory = cursor.fetchall()
    cursor.execute('SELECT id, name FROM MaterialType')
    materials = cursor.fetchall()
    conn.close()
    return render_template('manage_inventory.html', inventory=inventory, materials=materials)


@app.route('/manage_category', methods=['GET', 'POST'])
def manage_category():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    if request.method == 'POST':
        if 'add' in request.form:
            name = request.form['name']
            description = request.form['description']
            cursor.execute('INSERT INTO Category (name, description) VALUES (?, ?)', (name, description))
            conn.commit()
            flash('Category added successfully!', 'success')
        elif 'drop' in request.form:
            category_id = request.form['category_id']
            cursor.execute('DELETE FROM Category WHERE id = ?', (category_id,))
            conn.commit()
            flash('Category dropped successfully!', 'success')
        return redirect(url_for('manage_category'))
    cursor.execute('SELECT * FROM Category')
    categories = cursor.fetchall()
    conn.close()
    return render_template('manage_category.html', categories=categories)

if __name__ == '__main__':
    app.run(debug=True)

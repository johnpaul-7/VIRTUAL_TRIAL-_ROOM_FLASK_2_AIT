import sqlite3
import random
import os
import csv
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, send_from_directory, session, url_for, redirect
from src.utils.all_utils import read_yaml, create_directory
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import time
import os
app = Flask(__name__)
app.secret_key = '\xf0?a\x9a\\\xff\xd4;\x0c\xcbHi'
import os
os.makedirs('static/profile_pictures', exist_ok=True)
def base64_to_image(base64_data):
    base64_data = base64_data.split(",")[-1]
    image_data = base64.b64decode(base64_data)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()
# Add this new command for profile table
command_profile = """
CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    full_name TEXT,
    email TEXT,
    phone TEXT,
    age INTEGER,
    gender TEXT,
    country TEXT,
    city TEXT,
    address TEXT,
    bio TEXT,
    profile_picture TEXT,
    preferences TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (username) REFERENCES user(name)
)
"""
cursor.execute(command_profile)
connection.commit()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

# Global variables
current_glasses = None
vs = None
detector = None
predictor = None
detector1 = None

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def init_camera():
    global vs, detector, predictor, detector1
    
    if detector is None:
        print("-> Loading the predictor and detector...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        detector1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if vs is None:
        print("-> Starting Video Stream")
        vs = cv2.VideoCapture(0)
        time.sleep(1.0)

def generate_frames():
    global current_glasses, vs, detector, predictor, detector1
    
    if current_glasses is None:
        current_glasses = "static/glasses/glass1.png"  # Default glasses
    
    while True:
        success, frame = vs.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            if len(rects) > 0:
                for (x, y, w, h) in rects:
                    try:
                        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                        eye = final_ear(shape)
                        ear = eye[0]
                        leftEye = eye[1]
                        rightEye = eye[2]

                        # Calculate eye centers
                        left_center = np.mean(leftEye, axis=0).astype(int)
                        right_center = np.mean(rightEye, axis=0).astype(int)

                        # Load and resize glasses
                        glass_image = cv2.imread(current_glasses, -1)
                        if glass_image is not None:
                            # Calculate the required dimensions for glasses
                            glass_width_resize = 2.5 * abs(right_center[0] - left_center[0])
                            scale_factor = glass_width_resize / glass_image.shape[1]
                            resize_glasses = cv2.resize(glass_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

                            # Determine position for the glasses overlay
                            glass_height, glass_width, _ = resize_glasses.shape
                            glass_x = int((left_center[0] + right_center[0]) / 2 - glass_width / 2)
                            glass_y = int(left_center[1] - glass_height / 2)

                            # Ensure the glasses are properly cropped if they extend beyond the frame boundaries
                            if glass_x < 0:
                                glass_x = 0
                            if glass_y < 0:
                                glass_y = 0
                            if glass_x + glass_width > frame.shape[1]:
                                glass_width = frame.shape[1] - glass_x
                            if glass_y + glass_height > frame.shape[0]:
                                glass_height = frame.shape[0] - glass_y

                            # Create overlay image with white background (EXACTLY LIKE YOUR ORIGINAL CODE)
                            overlay_image = np.ones(frame.shape, np.uint8) * 255
                            overlay_image[int(glass_y): int(glass_y + resize_glasses.shape[0]),
                                        int(glass_x): int(glass_x + resize_glasses.shape[1])] = resize_glasses

                            # Create mask from overlay image
                            gray_overlay = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)
                            _, mask = cv2.threshold(gray_overlay, 127, 255, cv2.THRESH_BINARY)

                            # Get background (original frame without glasses area)
                            background = cv2.bitwise_and(frame, frame, mask=mask)

                            # Invert mask
                            mask_inv = cv2.bitwise_not(mask)

                            # Extract glasses from overlay
                            glasses = cv2.bitwise_and(overlay_image, overlay_image, mask=mask_inv)

                            # Combine background and glasses
                            final_image = cv2.add(background, glasses)
                            
                            # Update frame with the final image
                            frame = final_image

                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def total_iems():
    return len(os.listdir("static/collections"))

config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#upload
upload_image_dir = artifacts['upload_image_dir']
uploadn_path = os.path.join(artifacts_dir, upload_image_dir)

# pickle_format_data_dir
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

#Feature path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)

#params_path
weight = params['base']['weights']
include_tops = params['base']['include_top']

#loading
feature_list = np.array(pickle.load(open(features_name,'rb')))
filenames = pickle.load(open(pickle_file,'rb'))


#model
model = ResNet50(weights= weight,include_top=include_tops,input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        create_directory(dirs=[uploadn_path])
        with open(os.path.join(uploadn_path,uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    items = os.listdir('static/data')
    files = random.sample(items, 12)
    return render_template('home.html', ti = total_iems(), files=files)

@app.route('/recommendation')
def recommendation():
    return render_template('recommend.html', ti = total_iems())

@app.route('/trial/<img>')
def trial(img):
    session['filename'] = 'static/data/'+img
    return render_template('trialroom.html', ti = total_iems())

@app.route('/cart')
def cart():
    List = []
    prices = []
    total_price = 0
    items = []
    
    for im in os.listdir("static/collections"):
        with open("prices.csv", "r") as f:
            reader = csv.reader(f)
            File = im.split('.')[0]
            for i in reader:
                if File in i:
                    price = float(i[1])
                    prices.append(i[1])
                    total_price += price
                    items.append({
                        "name": File,
                        "image": "http://127.0.0.1:5000/static/collections/"+im,
                        "price": price
                    })
                    List.append("http://127.0.0.1:5000/static/collections/"+im)
    
    return render_template('cart.html', 
                         ti=len(items), 
                         items=items,  # Pass items list with details
                         List=List, 
                         prices=prices, 
                         n=len(List),
                         total_price=total_price)


@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    data = request.json
    item_name = data.get('item_name')
    
    # Remove the file from collections directory
    try:
        # Find the image file
        for filename in os.listdir("static/collections"):
            if item_name in filename:
                os.remove(os.path.join("static/collections", filename))
                break
        
        return jsonify({'success': True, 'message': 'Item removed from cart'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/profile')
def profile():
    # Get current user from session (you'll need to implement proper session management)
    # For now, let's assume we get the username from session
    if 'user' in session:
        username = session['user']
        
        # Try to get existing profile data
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        cursor.execute("SELECT * FROM user_profile WHERE username = ?", (username,))
        profile_data = cursor.fetchone()
        
        # If profile exists, parse it
        profile_dict = {}
        if profile_data:
            columns = [description[0] for description in cursor.description]
            profile_dict = dict(zip(columns, profile_data))
        
        # Get cart count
        cart_count = total_iems()
        
        return render_template('profile.html', ti=cart_count, profile=profile_dict)
    else:
        # Redirect to login if not logged in
        return redirect(url_for('index'))

@app.route('/save_profile', methods=['POST'])
def save_profile():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})
    
    try:
        username = session['user']
        data = request.json
        
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        # Check if profile already exists
        cursor.execute("SELECT id FROM user_profile WHERE username = ?", (username,))
        existing_profile = cursor.fetchone()
        
        # Prepare profile data
        profile_data = {
            'username': username,
            'full_name': data.get('fullName', ''),
            'email': data.get('email', ''),
            'phone': data.get('phone', ''),
            'age': data.get('age'),
            'gender': data.get('gender', ''),
            'country': data.get('country', ''),
            'city': data.get('city', ''),
            'address': data.get('address', ''),
            'bio': data.get('bio', ''),
            'preferences': data.get('preferences', '{}'),
            'profile_picture': data.get('profilePicture', '')
        }
        
        if existing_profile:
            # Update existing profile
            update_query = """
            UPDATE user_profile SET 
            full_name = ?, email = ?, phone = ?, age = ?, gender = ?, 
            country = ?, city = ?, address = ?, bio = ?, 
            profile_picture = ?, preferences = ?, updated_at = CURRENT_TIMESTAMP
            WHERE username = ?
            """
            cursor.execute(update_query, (
                profile_data['full_name'],
                profile_data['email'],
                profile_data['phone'],
                profile_data['age'],
                profile_data['gender'],
                profile_data['country'],
                profile_data['city'],
                profile_data['address'],
                profile_data['bio'],
                profile_data['profile_picture'],
                profile_data['preferences'],
                username
            ))
        else:
            # Insert new profile
            insert_query = """
            INSERT INTO user_profile 
            (username, full_name, email, phone, age, gender, country, city, address, bio, profile_picture, preferences)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_query, (
                profile_data['username'],
                profile_data['full_name'],
                profile_data['email'],
                profile_data['phone'],
                profile_data['age'],
                profile_data['gender'],
                profile_data['country'],
                profile_data['city'],
                profile_data['address'],
                profile_data['bio'],
                profile_data['profile_picture'],
                profile_data['preferences']
            ))
        
        connection.commit()
        connection.close()
        
        return jsonify({'success': True, 'message': 'Profile saved successfully'})
        
    except Exception as e:
        print(f"Error saving profile: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/upload_profile_picture', methods=['POST'])
def upload_profile_picture():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})
    
    try:
        if 'profile_picture' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['profile_picture']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        username = session['user']
        
        # Create profile_pictures directory if it doesn't exist
        profile_pic_dir = 'static/profile_pictures'
        os.makedirs(profile_pic_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"{username}_{int(time.time())}.jpg"
        filepath = os.path.join(profile_pic_dir, filename)
        
        # Save the file
        file.save(filepath)
        
        # Save the file path to database
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        cursor.execute("""
        UPDATE user_profile SET profile_picture = ?, updated_at = CURRENT_TIMESTAMP 
        WHERE username = ?
        """, (filepath, username))
        
        connection.commit()
        connection.close()
        
        return jsonify({
            'success': True, 
            'message': 'Profile picture uploaded successfully',
            'filepath': filepath
        })
        
    except Exception as e:
        print(f"Error uploading profile picture: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/clear_cart', methods=['POST'])
def clear_cart():
    try:
        # Remove all files from collections directory
        for filename in os.listdir("static/collections"):
            file_path = os.path.join("static/collections", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return jsonify({'success': True, 'message': 'Cart cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
# Update your userlog route to set session
@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = ? AND password = ?"
        cursor.execute(query, (name, password))
        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            # Set session
            session['user'] = name
            return redirect(url_for('home'))

    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/imagetest', methods=['GET', 'POST'])
def imagetest():
    if request.method == 'POST':
        fileName = session['filename']
        fileName1 = request.form['img2']
        File = fileName.split('.')[0]
        File = File.replace('static/data/', '')
        fileName1 = "static/test_img/"+fileName1

        os.system(f"python detection.py --input_image {fileName1} --input_cloth {fileName}")

        f = open("prices.csv", "r")
        reader = csv.reader(f)
        for i in reader:
            if File in i:
                name = i[1]
        import random
        random_range = random.randint(80, 90)
        return render_template('trialroom.html', ti = total_iems(), dress=fileName, price=name, image=fileName1, output="static/result/"+os.listdir("static/result")[0],random_range=random_range)

    return render_template('recommend.html' , ti = total_iems())

@app.route('/livetest', methods=['GET', 'POST'])
def livetest():
    if request.method == 'POST':
        fileName = session['filename']
        filedata = request.form['img2']

        dlist = os.listdir('static/testpicture')
        for item in dlist:
            os.remove("static/testpicture/"+item)
        
        name1 = str(random.randint(1000, 9999))
        result_image = base64_to_image(filedata)
        result_image.save('static/testpicture/'+name1+'.png')

        File = fileName.split('.')[0]
        File = File.replace('static/data/', '')
        fileName = fileName
        fileName1 = 'static/testpicture/'+name1+'.png'

        os.system(f"python detection.py --input_image {fileName1} --input_cloth {fileName}")

        f = open("prices.csv", "r")
        reader = csv.reader(f)
        for i in reader:
            if File in i:
                name = i[1]

        return render_template('trialroom.html', ti = total_iems(),  dress=fileName, price=name, image=fileName1, output="static/result/"+os.listdir("static/result")[0])

    return render_template('recommend.html', ti = total_iems())

##@app.route('/uploads/<filename>')
##def uploaded_file(filename):
##    return send_from_directory(uploadn_path, filename)

@app.route('/choose', methods=['POST'])
def choose():
    Type = request.form['Type']
    session['type'] = Type
    if Type == 'small':
        images = []
        names = []
        for img in os.listdir('static/artifacts/SMALL'):
            images.append('static/artifacts/SMALL/'+img)
            names.append(img)
        return render_template('recommend.html', names=names, images=images, Type=Type, n=len(images))
    if Type == 'medium':
        images = []
        names = []
        for img in os.listdir('static/artifacts/MEDIUM'):
            images.append('static/artifacts/MEDIUM/'+img)
            names.append(img)
        return render_template('recommend.html', names=names, images=images, Type=Type, n=len(images))
    if Type == 'large':
        images = []
        names = []
        for img in os.listdir('static/artifacts/LARGE'):
            images.append('static/artifacts/LARGE/'+img)
            names.append(img)
        return render_template('recommend.html', names=names, images=images, Type=Type, n=len(images))
    if Type == 'x-large':
        images = []
        names = []
        for img in os.listdir('static/artifacts/EXTRA_LARGE'):
            images.append('static/artifacts/EXTRA_LARGE/'+img)
            names.append(img)
        return render_template('recommend.html', names=names, images=images, Type=Type, n=len(images))
    return render_template('recommend.html')

@app.route('/upload/<name1>')
def upload(name1):
    print(name1)
    if session['type'] == 'small':
        filename = 'SMALL\\'+name1
    if session['type'] == 'medium':
        filename = 'MEDIUM\\'+name1
    if session['type'] == 'large':
        filename = 'LARGE\\'+name1
    if session['type'] == 'x-large':
        filename = 'EXTRA_LARGE\\'+name1
        
    features = feature_extraction(os.path.join(artifacts_dir, filename), model)
    indices = recommend(features, feature_list)
    result = []
    for i in filenames:
        result.append(i.replace("data\\", ''))
    return render_template('recommend.html', ti = total_iems(), filenames=result, indices=indices[0])
@app.route('/glass')
def index1():
    # Initialize camera when first accessing the page
    init_camera()
    
    # Get list of glasses from the glasses folder
    glasses_folder = 'static/glasses'
    glasses_list = []
    if os.path.exists(glasses_folder):
        glasses_list = [f for f in os.listdir(glasses_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    return render_template('index1.html', glasses_list=glasses_list)

@app.route('/video_feed')
def video_feed():
    init_camera()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_glasses', methods=['POST'])
def select_glasses():
    global current_glasses
    glasses_name = request.json['glasses']
    current_glasses = f"static/glasses/{glasses_name}"
    print(f"Selected glasses: {current_glasses}")
    return jsonify({'status': 'success'})

@app.route('/start_camera')
def start_camera():
    init_camera()
    return jsonify({'status': 'camera_started'})

@app.route('/stop_camera')
def stop_camera():
    global vs
    if vs is not None:
        vs.release()
        vs = None
    cv2.destroyAllWindows()
    return jsonify({'status': 'camera_stopped'})  
@app.route('/get_profile_data')
def get_profile_data():
    if 'user' not in session:
        return jsonify({})
    
    try:
        username = session['user']
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        cursor.execute("SELECT * FROM user_profile WHERE username = ?", (username,))
        profile_data = cursor.fetchone()
        
        if profile_data:
            columns = [description[0] for description in cursor.description]
            profile_dict = dict(zip(columns, profile_data))
            return jsonify(profile_dict)
        
        return jsonify({})
        
    except Exception as e:
        print(f"Error getting profile data: {e}")
        return jsonify({})  
@app.route('/save_preferences', methods=['POST'])
def save_preferences():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})
    
    try:
        username = session['user']
        data = request.json
        preferences = data.get('preferences', '{}')
        
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        cursor.execute("""
        UPDATE user_profile SET preferences = ?, updated_at = CURRENT_TIMESTAMP 
        WHERE username = ?
        """, (preferences, username))
        
        connection.commit()
        connection.close()
        
        return jsonify({'success': True, 'message': 'Preferences saved successfully'})
        
    except Exception as e:
        print(f"Error saving preferences: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/update_security', methods=['POST'])
def update_security():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})
    
    try:
        username = session['user']
        data = request.json
        
        # Here you would implement actual password change logic
        # This is just a placeholder
        
        # Update two-factor authentication setting
        two_factor_auth = data.get('twoFactorAuth', False)
        
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        # In a real app, you'd store this in a separate security table
        # For now, just acknowledge the request
        
        connection.close()
        
        return jsonify({
            'success': True, 
            'message': 'Security settings updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating security: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

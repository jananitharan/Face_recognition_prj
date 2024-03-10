import cv2
import csv
from datetime import datetime
import face_recognition
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_user_folder(username):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], username)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for user_folder in os.listdir(app.config['UPLOAD_FOLDER']):
        user_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], user_folder)

        if os.path.isdir(user_folder_path):
            image_files = [f for f in os.listdir(user_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for image_file in image_files:
                image_path = os.path.join(user_folder_path, image_file)
                image = face_recognition.load_image_file(image_path)

                # Ensure there is at least one face encoding before accessing it
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encoding = face_encodings[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(user_folder)

    return known_face_encodings, known_face_names


known_face_encodings, known_face_names = load_known_faces()

@app.route('/recognize', methods=['GET'])
def recognize():

    video_capture = cv2.VideoCapture(0)

    # Specify the directory where you want to save the CSV file
    csv_directory = "C://Users/janan/PycharmProjects/pythonProjectdemo/"

    # Create CSV file for logging
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    csv_filename = os.path.join(csv_directory, f"{current_date}_log.csv")

    students = known_face_names.copy()
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Timestamp', 'Date']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            # Find all face locations and face encodings in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Loop through each face found in the frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Check if the face matches any of the known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match is found, use the name of the known face
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                    # Log the recognized person's name and timestamp into CSV
                    if name in students:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(name, " - Present")
                        students.remove(name)
                        csv_writer.writerow({'Name': name, 'Timestamp': timestamp, 'Date': now})

                # Draw rectangle and label around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user-provided name and uploaded image file
        username = request.form['username']
        uploaded_file = request.files['image']

        if username and uploaded_file:
            # Create a folder for the user with the provided name
            user_folder = create_user_folder(username)

            # Save the uploaded image to the user's folder
            filename = secure_filename(uploaded_file.filename)
            image_path = os.path.join(user_folder, filename)
            uploaded_file.save(image_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

import face_recognition
import numpy as np
import os
import cv2

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

root_directory = "D:/Workspace/AI/deeplearning/Image/"
# Duyệt qua từng thư mục con
for directory_name in os.listdir(root_directory):
    # Tạo đường dẫn đến thư mục con
    sub_directory = os.path.join(root_directory, directory_name)
    
    # Kiểm tra xem nó có phải là một thư mục
    if os.path.isdir(sub_directory):
        # Duyệt qua từng tệp ảnh trong thư mục con
        for image_name in os.listdir(sub_directory):
            # Kiểm tra định dạng tệp ảnh (ví dụ: jpg, png)
            if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                
                # Đường dẫn đầy đủ đến tệp ảnh
                image_path = os.path.join(sub_directory, image_name)
                print(image_path)
                img = face_recognition.load_image_file(image_path)
                arr = face_recognition.face_encodings(img)
                if (arr != []):
                    img_encoding = arr[0]
                    known_face_encodings.append(img_encoding)
                    known_face_names.append(directory_name)
            else:
                print("Tệp không phải ảnh:", image_name)

print('Learned encoding for', len(known_face_encodings), 'images.')
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    # process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
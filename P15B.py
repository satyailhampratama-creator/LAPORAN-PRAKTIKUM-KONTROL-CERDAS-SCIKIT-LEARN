import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATA_DIR = os.path.join(os.path.expanduser("~"), "Documents", "python", "buku", "pelatihan", "DATA")
data = []
labels = []
for dir_ in os.listdir(DATA_DIR):

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))
                    data.append(data_aux)
                    labels.append(dir_)

# Simpan file
save_path = os.path.join(DATA_DIR, 'data.pickle')

with open(save_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data berhasil disimpan!")
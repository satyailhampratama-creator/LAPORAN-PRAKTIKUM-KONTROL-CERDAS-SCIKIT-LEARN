import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open(
    r'C:\Users\Satya Ilham\Documents\python\buku\pelatihan\data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.1, shuffle=True, stratify=labels)
# Buat model
model = RandomForestClassifier()
# Training
model.fit(x_train, y_train)
# Testing
y_pred = model.predict(x_test)
score = accuracy_score(y_pred, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))
# Simpan model
with open('C:\\Users\\Documents\\python\\buku\\pelatihan\\model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model berhasil disimpan!")
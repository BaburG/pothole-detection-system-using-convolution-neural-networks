import numpy as np
import cv2
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

global size
size = 100

# Load the model with weights from the .keras file
model = load_model('sample.keras')

# Load Testing data: non-pothole
nonPotholeTestImages = glob.glob("C:/Users/Babur/Desktop/pothole-detection-system-using-convolution-neural-networks/My Dataset/test/Plain/*.*")
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]
test2 = [cv2.resize(img, (size, size)) for img in test2]
temp4 = np.asarray(test2)

# Load Testing data: potholes
potholeTestImages = glob.glob("C:/Users/Babur/Desktop/pothole-detection-system-using-convolution-neural-networks/My Dataset/test/Pothole/*.*")
test1 = [cv2.imread(img, 0) for img in potholeTestImages]
test1 = [cv2.resize(img, (size, size)) for img in test1]
temp3 = np.asarray(test1)

# Combine the test data
X_test = np.concatenate((temp3, temp4), axis=0)

# Reshape for model input
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

# Create labels for the test data
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)

y_test = np.concatenate((y_test1, y_test2), axis=0)

# Convert labels to categorical format
y_test = to_categorical(y_test)

# Predict classes for test data
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Print predicted results
for i in range(len(X_test)):
    print(f">>> Predicted={predicted_classes[i]}")

# Optional: Evaluate the model on the test set
metrics = model.evaluate(X_test, y_test)
for metric_name, metric_value in zip(model.metrics_names, metrics):
    print(f'{metric_name}: {metric_value}')

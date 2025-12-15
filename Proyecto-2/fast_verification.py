
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 1 # Just 1 epoch to verify pipeline
DATA_DIR = "dataset"
MAX_IMAGES_PER_CLASS = 50 # Fast mode

def load_data():
    images = []
    labels = []
    classes = os.listdir(DATA_DIR)
    print(f"Classes found: {classes}")
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Loading {class_name}...")
        count = 0
        for filename in os.listdir(class_dir):
            if count >= MAX_IMAGES_PER_CLASS:
                break
            img_path = os.path.join(class_dir, filename)
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = resize(image, (IMG_SIZE, IMG_SIZE), anti_aliasing=True, preserve_range=True)
                images.append(image)
                labels.append(i)
                count += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    return np.array(images), np.array(labels), classes

print("Loading data (fast mode)...")
X, Y, class_names = load_data()
print(f"Data loaded. Shape: {X.shape}")

# Preprocess
X = preprocess_input(X)
Y_one_hot = to_categorical(Y, num_classes=len(class_names))

# Split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)
print(f"Train shape: {train_X.shape}, Test shape: {test_X.shape}")

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False 

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.6)) 
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# Train (Briefly)
print("Starting fast training...")
model.fit(
    train_X, train_Y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_X, test_Y),
    verbose=1
)

# Verify Visualization Logic (Save a plot)
print("Verifying visualization logic...")
predicted_classes = np.argmax(model.predict(test_X), axis=1)
true_classes = np.argmax(test_Y, axis=1)

# Pick first test image to visualize
if len(test_X) > 0:
    idx = 0
    img = test_X[idx]
    # The fix: rescale from [-1, 1] to [0, 1]
    img_display = (img + 1) / 2
    
    plt.figure()
    plt.imshow(img_display)
    plt.title(f"True: {class_names[true_classes[idx]]}, Pred: {class_names[predicted_classes[idx]]}")
    plt.savefig("verification_plot.png")
    print("Verification plot saved to verification_plot.png")
else:
    print("No test data to visualize.")

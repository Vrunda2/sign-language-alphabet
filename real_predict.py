import cv2
import json
import numpy as np
import tensorflow as tf

# Load trained model
print("Loading model...")
model = tf.keras.models.load_model("model/asl_model.h5")

# Load class indices
with open('model/class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Create index to class name mapping
index_to_class = {v: k for k, v in class_indices.items()}
class_labels = [index_to_class[i] for i in range(len(index_to_class))]

print("Class labels:", class_labels)
print("Model loaded successfully!")

IMG_SIZE = 64

def preprocess(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    # Normalize
    img = img.astype("float32") / 255.0
    # Add batch dimension
    return np.expand_dims(img, axis=0)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📷 ASL Recognition Started!")
print("📷 Position your hand in the green rectangle")
print("📷 Press 'q' to quit")

# Prediction smoothing
prediction_history = []
history_size = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    # Define ROI (Region of Interest)
    x1, y1 = int(width * 0.1), int(height * 0.1)
    x2, y2 = int(width * 0.6), int(height * 0.6)
    roi = frame[y1:y2, x1:x2]

    # Make prediction
    input_image = preprocess(roi)
    prediction = model.predict(input_image, verbose=0)[0]
    
    # Smooth predictions
    prediction_history.append(prediction)
    if len(prediction_history) > history_size:
        prediction_history.pop(0)
    
    avg_prediction = np.mean(prediction_history, axis=0)
    class_idx = np.argmax(avg_prediction)
    confidence = avg_prediction[class_idx]

    # Display prediction
    if confidence > 0.6:  # High confidence threshold
        label = f"{class_labels[class_idx]} ({confidence*100:.1f}%)"
        color = (0, 255, 0)  # Green
    elif confidence > 0.3:  # Medium confidence
        label = f"{class_labels[class_idx]} ({confidence*100:.1f}%)"
        color = (0, 255, 255)  # Yellow
    else:
        label = "Show clear sign"
        color = (0, 0, 255)  # Red

    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw prediction
    cv2.putText(frame, label, (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Show top 3 predictions (for debugging)
    top_3_idx = np.argsort(avg_prediction)[-3:][::-1]
    for i, idx in enumerate(top_3_idx):
        debug_text = f"{class_labels[idx]}: {avg_prediction[idx]*100:.1f}%"
        cv2.putText(frame, debug_text, (x1, y2 + 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display frames
    cv2.imshow("ASL Recognition", frame)
    
    # Show ROI separately
    roi_display = cv2.resize(roi, (200, 200))
    cv2.imshow("Hand Region", roi_display)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
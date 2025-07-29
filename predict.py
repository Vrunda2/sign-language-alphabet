import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/asl_model.h5")
labels = list(model.class_names) if hasattr(model, "class_names") else list(range(model.output_shape[-1]))

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Predicted: {labels[class_idx]} with confidence {confidence*100:.2f}%")

# Example usage:
# predict_image("dataset/A/image_1.jpg")

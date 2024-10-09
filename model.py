import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_emotion(input_img, model_path='emotion_classification_model.h5', img_height=128, img_width=128):
    """
    Predicts the emotion from an input image, either from an image path or a frame captured via OpenCV (webcam).
    """
    # Load the trained model
    model = load_model(model_path)
    
    # Check if input is a path (string) or an image array (OpenCV image)
    if isinstance(input_img, str):
        # Load and preprocess the image from the path
        img = image.load_img(input_img, target_size=(img_height, img_width))
        img_array = image.img_to_array(img) / 255.0
    else:
        # Assume input is an image array (OpenCV image from webcam)
        img_array = cv2.resize(input_img, (img_height, img_width))  # Resize image
        img_array = np.array(img_array, dtype="float32") / 255.0  # Normalize the image
    
    # Add batch dimension for model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class of the image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Define the fixed class indices for emotion labels
    class_indices = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}
    
    # Get the emotion label corresponding to the predicted class
    emotion_label = class_indices.get(predicted_class, "Unknown")
    
    return emotion_label

def process_image_or_webcam(image_path=None):
    """
    Function to either load an image from the provided image path or capture from a webcam.
    """
    if image_path:
        # If an image path is provided, use it
        emotion = predict_emotion(image_path)
        print(f'Predicted emotion (from image path): {emotion}')
    else:
        # If no image path is provided, start webcam capture
        cap = cv2.VideoCapture(0)  # 0 is the default webcam
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        while True:
            # Capture frame-by-frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Predict the emotion from the webcam frame
            emotion = predict_emotion(frame)
            print(f'Predicted Emotion: {emotion}')

            # Display the current frame
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Webcam - Press Q to quit', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the capture and close the window
        cap.release()
        cv2.destroyAllWindows()


#  image path
image_path = "C:\\Users\\USER\\Pictures\\Screenshots\\Screenshot 2024-10-07 125258.png"
process_image_or_webcam(image_path=image_path)

#  capture from the webcam
#process_image_or_webcam()  # No image path provided, so webcam will be used

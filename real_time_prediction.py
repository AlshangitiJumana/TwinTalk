import os
import cv2
import mediapipe as mp
import torch
import time  # ✅ Import time for delay
from playsound import playsound
from gtts import gTTS
from dotenv import load_dotenv

from live_video_processing import preprocess_live
from load_and_predict import classify_live, load_model
from utils import load_class_names

load_dotenv()

def continuous_capture_and_classify():
    cap = cv2.VideoCapture(0)  # Open the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera not accessible!")
        return

    model = load_model().to("cpu")  # Load the trained model
    idx2label = load_class_names()  # Load the sign language labels

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    print("Show your hands to the camera to start predictions.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hands_detected = False  # Reset hand detection flag

        # Draw detected hands on the screen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            hands_detected = True  # Hands detected

        cv2.imshow("Sign Language to Speech", frame)

        # Only predict if hands are detected
        if hands_detected:
            input_tensor = preprocess_live(cap)  # Process the live video frames
            if input_tensor is not None:
                predictions = classify_live(input_tensor, model, idx2label)
                print("Predictions:", predictions)

                # Convert the prediction to speech
                top_label = predictions[0]
                tts = gTTS(text=top_label, lang='en')
                tts_file = 'prediction.mp3'
                tts.save(tts_file)
                playsound(tts_file)

                # ✅ Add a 2-second delay before detecting the next sign
                print("⏳ Waiting for 2 seconds before the next sign...")
                time.sleep(2)  # Pause for 2 seconds

        # Quit program when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    continuous_capture_and_classify()

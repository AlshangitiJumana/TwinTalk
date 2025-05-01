# import os
# import cv2
# import mediapipe as mp
# import torch
# import time  # ✅ Import time for delay
# from playsound import playsound
# from gtts import gTTS
# from dotenv import load_dotenv

# from live_video_processing import preprocess_live
# from load_and_predict import classify_live, load_model
# from utils import load_class_names

# load_dotenv()

# def continuous_capture_and_classify():
#     cap = cv2.VideoCapture(0)  # Open the webcam
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduce height

#     if not cap.isOpened():
#         print("❌ Camera not accessible!")
#         return

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_model().to(device)  # Load the trained model
#     idx2label = load_class_names()  # Load the sign language labels

#     # Initialize MediaPipe Hands
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

#     print("Show your hands to the camera to start predictions.")
#     frame_count = 0
#     results = None  # ✅ Initialize results before entering the loop

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Failed to capture frame.")
#             break

#         frame_count += 1

#         # ✅ Run hand detection every 5 frames
#         if frame_count % 5 == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(frame_rgb)  # ✅ Now results is always assigned

#         hands_detected = False  # Reset hand detection flag

#         # ✅ Check if results is valid before accessing it
#         if results and results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
#             hands_detected = True  # Hands detected

#         cv2.imshow("Sign Language to Speech", frame)

#         # Only predict if hands are detected
#         if hands_detected:
#             input_tensor = preprocess_live(cap)  # Process the live video frames
#             if input_tensor is not None:
#                 predictions = classify_live(input_tensor, model, idx2label)
#                 print("Predictions:", predictions)

#                 # Convert the prediction to speech
#                 top_label = predictions[0]
#                 tts = gTTS(text=top_label, lang='en')
#                 tts_file = 'prediction.mp3'
#                 tts.save(tts_file)
#                 playsound(tts_file)

#                 # ✅ Add a 2-second delay before detecting the next sign
#                 print("⏳ Waiting for 2 seconds before the next sign...")
#                 time.sleep(2)  # Pause for 2 seconds

#         # Quit program when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     continuous_capture_and_classify()


import os
import cv2
import mediapipe as mp
import torch
import time
import threading
from playsound import playsound
from gtts import gTTS
from dotenv import load_dotenv

from live_video_processing import preprocess_live
from load_and_predict import classify_live, load_model
from utils import load_class_names

load_dotenv()

# ✅ Load model on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)
idx2label = load_class_names()  # ✅ Load the sign language labels (before using it)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # ✅ Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)  # ✅ Reduce FPS to 10

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

frame_count = 0
print("Show your hands to the camera to start predictions.")

def predict_in_background(input_tensor, model, idx2label):
    prediction_result = classify_live(input_tensor.to(device), model, idx2label)

    # ✅ Ensure prediction_result is a tuple (word, confidence)
    if not isinstance(prediction_result, tuple) or len(prediction_result) != 2:
        print("❌ Invalid prediction result, skipping...")
        return

    word, confidence = prediction_result  # ✅ Extract word and confidence
    print(f"🔊 Predictions: {word} (Confidence: {confidence:.2f})")

    # ✅ Correct condition to skip low-confidence predictions
    CONFIDENCE_THRESHOLD = 0.7  # ✅ Set the threshold to 0.75
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"❌ Low confidence ({confidence:.2f}), skipping speech output.")
        return  # ✅ Skip playing the word

    # # ✅ Convert the prediction to speech if confidence is high enough
    # tts = gTTS(text=word, lang='en')
    # tts_file = 'prediction.mp3'
    # tts.save(tts_file)
        # ✅ Convert the chosen word to speech
    tts = gTTS(text=word, lang='en')
    tts_file = 'prediction.mp3'
    tts.save(tts_file)
    playsound(tts_file)
    os.remove(tts_file)

    print(f"🔊 Playing word: {word}")  # ✅ Debug message

    # ✅ Play the sound
  # ✅ Delete after playing
    time.sleep(2)  # ✅ Wait before next prediction

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        frame_count += 1
        if frame_count % 5 != 0:  # ✅ Skip frames for performance boost
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hands_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            hands_detected = True

        cv2.imshow("Sign Language to Speech", frame)

        if hands_detected:
            input_tensor = preprocess_live(cap)
            if input_tensor is not None:
                prediction_thread = threading.Thread(target=predict_in_background, args=(input_tensor, model, idx2label))
                prediction_thread.start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Exiting Program...")

cap.release()
cv2.destroyAllWindows()

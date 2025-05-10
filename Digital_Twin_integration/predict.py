import sys
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from transformers import pipeline

# === Language Model Setup (using flan-t5-large) ===
nlp = pipeline("text2text-generation", model="google/flan-t5-large", framework="pt")

def generate_sentence_better(accepted_words):
    if not accepted_words:
        return ""
    prompt = f"Please write a grammatically correct and meaningful sentence that includes all of the following words: {', '.join(accepted_words)}."
    output = nlp(prompt, max_length=50, do_sample=True, top_k=50, temperature=0.9)[0]['generated_text']
    return output

# === LSTM Model Definition ===
class SignLanguageLSTM(nn.Module):
    def __init__(self, num_classes, input_size=126, hidden_size=128, dropout=0.5):
        super(SignLanguageLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(p=dropout)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.dropout_lstm = nn.Dropout(p=dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(hidden_size * 2)
        self.dropout_fc = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout_conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.norm1(x)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)
        x = self.norm2(x)
        x = torch.max(x, dim=1)[0]
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# === Model Config ===
MODEL_WEIGHTS = r".\The_LSTM_model_wights_path.pth"
NUM_CLASSES = 90
CLASS_NAMES = [
    'all', 'almost', 'approve', 'before', 'boss', 'break', 'business', 'busy', 'but', 'buy', 'can',
    'change', 'clock', 'computer', 'deaf', 'decide', 'delay', 'different', 'discuss', 'drink',
    'eat', 'email', 'evaluate', 'explain', 'family', 'fine', 'finish', 'forget', 'full', 'give',
    'goal', 'have', 'hearing', 'help', 'how', 'idea', 'improve', 'inform', 'last', 'later', 'leader',
    'like', 'manager', 'many', 'meet', 'meeting', 'money', 'month', 'need', 'no', 'now', 'office',
    'paper', 'plan', 'policy', 'presentation', 'problem', 'professional', 'provide', 'responsibility',
    'result', 'role', 'same', 'schedule', 'secretary', 'sell', 'show', 'sorry', 'study', 'support',
    'table', 'take', 'team', 'time', 'trade', 'understand', 'vacation', 'visit', 'wait', 'want',
    'week', 'what', 'who', 'why', 'with', 'work', 'workshop', 'year', 'yes', 'yesterday'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageLSTM(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model = model.to(device)
model.eval()

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Prediction Function ===
def predict_sign_from_base64_frames(frames_b64):
    all_landmarks = []

    for frame_b64 in frames_b64:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]

        decoded = base64.b64decode(frame_b64)
        image = Image.open(BytesIO(decoded)).convert("RGB")
        rgb_frame = np.array(image)

        results = hand_detector.process(rgb_frame)

        hand_features = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                single_hand = []
                for lm in hand_landmarks.landmark:
                    single_hand.extend([lm.x, lm.y, lm.z])
                hand_features.append(single_hand)

        if len(hand_features) == 1:
            hand_features.append([0.0] * 63)
        elif len(hand_features) == 0:
            hand_features = [[0.0] * 63, [0.0] * 63]

        features = np.array(hand_features[0] + hand_features[1])
        all_landmarks.append(features)

    if not all_landmarks:
        return None

    data_tensor = torch.tensor(all_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(data_tensor)
        pred_class = output.argmax(dim=1).item()

    predicted_text = CLASS_NAMES[pred_class]
    return predicted_text

# === Entry Point ===
if __name__ == '__main__':
    input_data = sys.stdin.read()
    data = json.loads(input_data)

    frames = data.get("frames", [])
    accepted_words = data.get("accepted_words", [])

    predicted_word = predict_sign_from_base64_frames(frames) if frames else None
    result = { "predicted_word": predicted_word }

    if isinstance(data, dict) and data.get("final") is True:
        accepted_words = data.get("accepted_words", [])
        result["enhanced_sentence"] = generate_sentence_better(accepted_words)

    print(json.dumps(result))

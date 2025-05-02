# Sign Language Translation with CNN_LSTM and Digital Twin Integration

This project implements a full pipeline for translating sign language videos into spoken language using deep learning and a virtual avatar. It combines landmark detection, graph-based action recognition, and avatar speech synthesis using the D-ID API.

---

## 📁 Project Structure

- `landmark_detection.ipynb`: Extracts hands landmarks from sign language videos and converts them into `.npy` format.
- `Train.ipynb`: Trains the CNN-LSTM model.
- `best_model/LSTM71.11.pth`: Contains the best CNN-LSTM model.
- `DT_integration.ipynb`: Integrates the translation output with the D-ID avatar API to generate digital twin.
- `Digital_Twin/server.js`: Backend service to connect with D-ID API. You can insert your API key here.

---

## 🚀 How to Run

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node.js dependencies:**
   ```bash
   cd Digital_Twin
   npm install
   ```

3. **Add your D-ID API key:**
   - Open `Digital_Twin/server.js`
   - Replace the existing API key string with your own.

4. **Run the backend server:**
   ```bash
   node server.js
   ```

5. **Run notebooks based on your use case:**

### ➤ Option 1: Use the provided model
- Skip training steps.
- Start directly with `DT_integration.ipynb`.

### ➤ Option 2: Train the model on your own dataset
- Run `landmark_detection.ipynb` to convert your videos into `.npy` landmark files (make sure to use your correct paths).
- Run `Train.ipynb` to train the CNN-LSTM model.
- Finally, run `DT_integration.ipynb` to integrate with the avatar.


---

## ✅ Output

The system detects sign language the camera, translates it using the CNN-LSTM model, and then generates spoken digtal twin.

---

## 📦 Model

Trained model is saved at:
```
best_model/LSTM71.11.pth
```

---

## 📌 Notes

- Ensure your D-ID account has access to the avatar API.
- Video input should be pre-processed into landmarks using MediaPipe.


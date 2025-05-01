# Sign Language Translation with ST-GCN and Digital Twin Integration

This project implements a full pipeline for translating sign language videos into spoken language using deep learning and a virtual avatar. It combines landmark detection, graph-based action recognition, and avatar speech synthesis using the D-ID API.

---

## 📁 Project Structure

- `landmark_detection.ipynb`: Extracts hands landmarks from sign language videos and converts them into `.npy` format.
- `Data_preparation_for_STGCN.ipynb`: Prepares the `.npy` data to be compatible with the ST-GCN model.
- `fine_tune.ipynb`: Fine-tunes the ST-GCN model for sign language translation using the prepared dataset.
- `Best_fine_tuned_model/st-gcn.pt`: Contains the best fine-tuned ST-GCN model.
- `Digital_Twin_Integration.ipynb`: Integrates the translation output with the D-ID avatar API to generate digital twin.
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

### ➤ Option 1: Use the provided pre-trained model
- Skip training steps.
- Start directly with `Digital_Twin_Integration.ipynb`.

### ➤ Option 2: Fine-tune the model on your own dataset
- Run `landmark_detection.ipynb` to convert your videos into `.npy` landmark files (make sure to use your correct paths).
- Run `Data_preparation_for_STGCN.ipynb` to format the data.
- Run `fine_tune.ipynb` to fine-tune the ST-GCN model.
- Finally, run `Digital_Twin_Integration.ipynb` to integrate with the avatar.


---

## ✅ Output

The system detects sign language the camera, translates it using the ST-GCN model, and then generates spoken digtal twin.

---

## 📦 Model

Trained model is saved at:
```
Best_fine_tuned_model/st-gcn.pt
```

---

## 📌 Notes

- Ensure your D-ID account has access to the avatar API.
- Video input should be pre-processed into landmarks using MediaPipe.


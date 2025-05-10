# 🧠 TwinTalk: Real-Time Sign Language Translation with Digital Twin Avatars

TwinTalk is an AI-powered pipeline that translates American Sign Language (ASL) into speech using a deep learning model (ST-GCN) and a virtual avatar generated with the D-ID API.

---

## 🗂️ Project Structure

```
TwinTalk/
├── server.js                    # Express.js server to handle API communication with D-ID
├── predict.py                  # Handles ST-GCN inference for translated gestures
├── public/                     # Frontend web interface for the translation system
│   ├── index.html              # Main UI
│   └── login.html              # Optional login screen (static)
├── package.json                # Node.js project metadata
├── package-lock.json           # Node.js dependency tree
```

---

## 🌐 Live Website (Optional)

> 🧪 If deployed locally, access the web app at:
>
> ```
> http://localhost:3000/
> ```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/TwinTalk.git
cd TwinTalk
```

### 2. Install Node.js dependencies

```bash
npm install
```

### 3. Add your D-ID API key

- Open the file `server.js`
- Replace the placeholder string with your actual D-ID API key:

```js
const apiKey = "YOUR_DID_API_KEY";
```

### 4. Start the backend server

```bash
node server.js
```

### 5. Launch the frontend

- Open your browser and go to:

```
http://localhost:3000/
```

---

## 📦 Requirements

- Node.js ≥ 14
- A valid [D-ID API Key](https://www.d-id.com/)
- Python ≥ 3.8 (optional, if you run `predict.py`)
- Trained ST-GCN model (`st-gcn.pt`)

---

## ✅ Output

The system captures a sign language video, processes the hand landmarks, translates the gesture using ST-GCN, and generates a speaking avatar using the D-ID API.

---

## 📌 Notes

- Be sure your D-ID account has API access enabled.
- Sign language landmarks should be extracted with [MediaPipe](https://google.github.io/mediapipe/).
- If you want to customize the avatar or language, modify the payload in `server.js`.

---

## 📃 License

This project is for educational and non-commercial use only. For other uses, please contact the project maintainer.

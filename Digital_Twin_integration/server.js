const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const port = 3000;
const API_KEY = 'Your_D-ID_API_Key'; // D-ID Pro API Key

let confirmedWords = [];
let lastPredictedWord = null;

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(express.static('public'));

// === /predict — Handle both prediction and final sentence generation ===
app.post('/predict', (req, res) => {
  const python = spawn('python', ['predict.py']);
  const input = JSON.stringify(req.body); // { frames, accepted_words?, final? }

  let result = '';
  python.stdin.write(input);
  python.stdin.end();

  python.stdout.on('data', (data) => result += data.toString());
  python.stderr.on('data', (data) => console.error('🐍 Python error:', data.toString()));

  python.on('close', () => {
    try {
      const parsed = JSON.parse(result);
      lastPredictedWord = parsed.predicted_word;
      res.json(parsed);
    } catch (e) {
      console.error('❌ Failed to parse Python output');
      res.status(500).json({ error: 'Prediction failed or invalid output.' });
    }
  });
});

// === /decision — Handle Accept, Retry, Finish ===
app.post('/decision', async (req, res) => {
  const { decision, word, add_before_finish } = req.body;

  if (decision === 'y') {
    confirmedWords.push(word);
    lastPredictedWord = null;
    return res.json({ new_word: null });
  }

  if (decision === 'n') {
    lastPredictedWord = null;
    return res.json({ new_word: null });
  }

  if (decision === 'f') {
    if (add_before_finish && word !== '-') {
      confirmedWords.push(word);
    }

    try {
      const sentence = await generateSentence(confirmedWords);
      confirmedWords = []; // reset
      return res.json({ enhanced_sentence: sentence });
    } catch (e) {
      console.error('❌ Sentence generation failed:', e.message);
      return res.status(500).json({ error: 'Failed to generate sentence.' });
    }
  }

  res.status(400).json({ error: 'Invalid decision type.' });
});

// === /generate — Send sentence to D-ID and return video URL ===
app.post('/generate', async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: 'No text provided.' });

  try {
    const videoUrl = await createAvatar(text);
    res.json({ videoUrl });
  } catch (e) {
    console.error('❌ D-ID API failed:', e.message);
    res.status(500).json({ error: 'Failed to create avatar video.' });
  }
});

// === Helper — Generate sentence via Python ===
async function generateSentence(words) {
  const python = spawn('python', ['predict.py']);
  const input = JSON.stringify({ frames: [], accepted_words: words, final: true });

  let result = '';
  python.stdin.write(input);
  python.stdin.end();

  return new Promise((resolve, reject) => {
    python.stdout.on('data', (data) => result += data.toString());
    python.on('close', () => {
      try {
        const parsed = JSON.parse(result);
        resolve(parsed.enhanced_sentence || words.join(' '));
      } catch (e) {
        reject(e);
      }
    });
  });
}

// === Helper — D-ID API Call ===
async function createAvatar(text) {
  const createRes = await axios.post(
    'https://api.d-id.com/clips',
    {
      presenter_id: 'frank-gvlo7vAP2C',
      script: {
        type: 'text',
        input: text,
        provider: {
          type: 'elevenlabs',
          voice_id: 'ErXwobaYiN019PkySvjV'
        }
      },
      config: { fluent: true, pad_audio: 0.2, stitch: true }
    },
    {
      headers: {
        Authorization: `Basic ${API_KEY}`,
        'Content-Type': 'application/json'
      }
    }
  );

  const clipId = createRes.data.id;
  for (let i = 0; i < 60; i++) {
    const statusRes = await axios.get(`https://api.d-id.com/clips/${clipId}`, {
      headers: { Authorization: `Basic ${API_KEY}` }
    });

    if (statusRes.data.result_url) return statusRes.data.result_url;
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  throw new Error('Timeout waiting for avatar video.');
}

// === Start Server ===
app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});

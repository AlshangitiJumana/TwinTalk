const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const cors = require('cors');
const app = express();
const port = 3000;

const API_KEY = 'PUT_YOUR_KEY_HERE';

app.use(cors());
app.use(bodyParser.json());
app.use(express.static('public'));

app.post('/generate', async (req, res) => {
  const { text } = req.body;

  try {
    console.log(`🟢 Request received for text: "${text}"`);

    const createRes = await axios.post(
      'https://api.d-id.com/talks',
      {
        script: {
          type: 'text',
          input: text,
          provider: {
            type: 'microsoft',
            voice_id: 'en-US-JennyNeural'
          },
          ssml: false
        },
        config: {
          fluent: true,
          pad_audio: 0.2,
          stitch: true
        },
        avatar_id: 'ava-4e9251c0-7c49-11ee-bf9e-acde48001122'
      },
      {
        headers: {
          Authorization: `Basic ${API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    const talkId = createRes.data.id;
    console.log(`🕒 Waiting for video to be ready (talk ID: ${talkId})...`);

    let resultUrl = null;
    for (let i = 0; i < 60; i++) {
      const statusRes = await axios.get(`https://api.d-id.com/talks/${talkId}`, {
        headers: {
          Authorization: `Basic ${API_KEY}`
        }
      });

      if (statusRes.data.result_url) {
        resultUrl = statusRes.data.result_url;
        console.log(`✅ Video ready: ${resultUrl}`);
        break;
      }

      console.log(`⏳ Not ready yet (${i + 1}/15)... retrying in 2s`);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }

    if (!resultUrl) {
      console.error('❌ Timeout: Video not ready after 30s.');
      return res.status(504).json({ error: 'Timeout: video not ready' });
    }

    res.json({ videoUrl: resultUrl });

  } catch (error) {
    console.error('D-ID API error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to generate video' });
  }
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});
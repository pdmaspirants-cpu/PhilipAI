import express from 'express';
import multer from 'multer';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Configure Multer for memory storage
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json());

// API Route for Transcription
app.post('/api/transcribe', upload.single('video'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'No video file provided' });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'GEMINI_API_KEY is not configured on the server' });
    }

    const ai = new GoogleGenAI({ apiKey });
    
    // Convert buffer to base64 for Gemini
    const base64Data = file.buffer.toString('base64');
    
    const response = await ai.models.generateContent({
      model: 'gemini-3.1-pro-preview', // Using the latest 3.1 Pro model
      contents: [
        {
          parts: [
            {
              inlineData: {
                data: base64Data,
                mimeType: file.mimetype
              }
            },
            {
              text: 'Listen to this Malayalam audio. Transcribe it, translate it into English, and output the result strictly as a valid SRT subtitle file with timestamps.'
            }
          ]
        }
      ]
    });

    const srtContent = response.text;
    if (!srtContent) {
      throw new Error('Gemini failed to generate SRT content');
    }

    res.json({ srt: srtContent });
  } catch (error: any) {
    console.error('Transcription Error:', error);
    res.status(500).json({ error: error.message || 'Internal Server Error' });
  }
});

// Vite middleware for development
async function setupVite() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    // Serve static files in production
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

setupVite();

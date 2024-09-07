import { Hono } from "hono";
import { logger } from "hono/logger";
import { serve } from '@hono/node-server';
import { WebSocketServer, WebSocket } from "ws";  // WebSocketServer and WebSocket imports
import twilio from "twilio";
import { google } from "googleapis";
import axios from "axios";
import { fileURLToPath } from "url";
import { dirname } from "path";
import OpenAI from 'openai';

// Fix for __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Initialize Google Sheets API
const GOOGLE_SHEET_ID = "10oAG2URrrX1YJr0StNAXszFycYPwBMmj9N6Y18TwcVk";
const GOOGLE_SHEET_RANGE = "Sheet1!A1:B1";

const SERVICE_ACCOUNT_INFO = {
  "type": "service_account",
  "project_id": "ferrous-thought-432910-d3",
  "private_key_id": "182b04aace7cf843455d7097555028be3a55566f",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n",
  "client_email": "leadgen@ferrous-thought-432910-d3.iam.gserviceaccount.com",
  "client_id": "116199920320342924096",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/leadgen%40ferrous-thought-432910-d3.iam.gserviceaccount.com"
};

const ELEVEN_LABS_API_KEY = process.env.ELEVEN_LABS_API_KEY;
const VOICE_ID = "6xp1pT27VobnUWUNT1jQ";
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Service account setup for Google Sheets
const auth = new google.auth.GoogleAuth({
  credentials: SERVICE_ACCOUNT_INFO,
  scopes: ["https://www.googleapis.com/auth/spreadsheets"],
});
const sheets = google.sheets({ version: "v4", auth });

// Initialize Hono app
const app = new Hono();
app.use("*", logger());

// Root path handler
app.get('/', (c) => {
  return c.text('Hello, Hono on Node.js!');
});

// Create WebSocket server
const wss = new WebSocketServer({ noServer: true });

// Handle WebSocket connections
wss.on('connection', (ws) => {
  console.log('WebSocket connection established');
  ws.on('message', (message) => {
    console.log('Received:', message);
    ws.send('Message received!');
  });
  ws.on('close', () => {
    console.log('WebSocket connection closed');
  });
});

// Upgrade HTTP request to WebSocket request
app.get('/stream-audio', async (c) => {
  const { req } = c;

  const server = req.raw.server;
  if (server) {
    server.on('upgrade', (request, socket, head) => {
      if (request.url === '/stream-audio') {
        wss.handleUpgrade(request, socket, head, (ws) => {
          wss.emit('connection', ws, request);
        });
      } else {
        socket.destroy();
      }
    });
  } else {
    return c.text('Upgrade failed: server is not defined', 500);
  }

  return c.text('WebSocket endpoint ready.');
});

// Function to handle Twilio incoming call
app.post("/incoming-call", (c) => {
  const voiceResponse = new twilio.twiml.VoiceResponse();

  // Connect Twilio call to a WebSocket for real-time audio streaming
  voiceResponse.connect().stream({
    url: "wss://ecc7f349-519e-4bdd-9cf9-e6f5697e81bc-00-2w0vh2l26wc9.spock.replit.dev/stream-audio",
  });

  c.header("Content-Type", "application/xml");
  return c.body(voiceResponse.toString());
});

// Async function to handle ElevenLabs streaming audio generation
const streamAudioToTwilio = async (text: string) => {
  const ws = new WebSocket('wss://ecc7f349-519e-4bdd-9cf9-e6f5697e81bc-00-2w0vh2l26wc9.spock.replit.dev/stream-audio');

  ws.on("open", async () => {
    console.log("WebSocket connection opened");

    // Request ElevenLabs Streaming API to generate real-time audio
    const elevenLabsStream = await axios({
      method: "POST",
      url: `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}/stream`,
      headers: {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "content-type": "application/json",
      },
      data: {
        text: text,
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.8,
        },
      },
      responseType: "stream",
    });

    // Send the generated audio stream to Twilio WebSocket
    elevenLabsStream.data.on("data", (chunk) => {
      ws.send(chunk);
    });

    elevenLabsStream.data.on("end", () => {
      ws.close();
    });
  });

  ws.on("error", (error) => {
    console.error("WebSocket error:", error);
  });
};

// Function to send data to Google Sheets
const sendToGoogleSheet = async (email: string, notes: string) => {
  const values = [[email, notes]];
  const resource = { values };

  try {
    const result = await sheets.spreadsheets.values.append({
      spreadsheetId: GOOGLE_SHEET_ID,
      range: GOOGLE_SHEET_RANGE,
      valueInputOption: "RAW",
      resource,
    });
    console.log("Data successfully sent to Google Sheets.");
    return true;
  } catch (error) {
    console.error(
      `Exception occurred while sending data to Google Sheets: ${error}`,
    );
    return false;
  }
};

// Function to handle session management and persona switching
app.post("/respond", async (c) => {
  try {
    const formData = await c.req.formData();
    const voiceInput = formData.get("SpeechResult")?.toString()!;
    const userId = "user_demo"; // Simulating a user session

    const userSessions: Record<string, any> = {}; // Assuming session object

    if (!userSessions[userId]) {
      userSessions[userId] = {
        exchangeCount: 0,
        infoCollected: false,
        conversationHistory: [],
      };
    }

    const userSession = userSessions[userId];
    userSession.exchangeCount += 1;

    // Fetch persona prompt based on session
    const personaPrompt = userSession.exchangeCount < 3
      ? `You are a representative of the company Coop, your role is to answer the caller with short answers, while subtly pitching the company Coop.`
      : "You are a friendly and persuasive salesman working for the company Coop.";

    // Add user input to conversation history
    userSession.conversationHistory.push({ role: "user", content: voiceInput });

    // Construct system message
    const systemMessage = { role: "system", content: personaPrompt };
    const messages = [systemMessage, ...userSession.conversationHistory];

    // Get assistant response from OpenAI
    const chatCompletion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: messages,
    });

    const assistantResponse = chatCompletion.choices[0].message.content;
    userSession.conversationHistory.push({
      role: "assistant",
      content: assistantResponse,
    });

    // Check if email is present in user's input
    const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/;
    const detectedEmail = voiceInput.match(emailRegex)?.[0];
    if (detectedEmail) {
      console.log(`Email detected: ${detectedEmail}`);
      const success = await sendToGoogleSheet(detectedEmail, assistantResponse);
      if (success) {
        console.log("Data sent to Google Sheets successfully.");
      }
    }

    // Stream audio from ElevenLabs directly to Twilio via WebSocket
    await streamAudioToTwilio(assistantResponse);

    return c.text("Processing completed.", 200);
  } catch (error) {
    console.error("Error in /respond route:", error);
    return c.text("Server error", 500);
  }
});

// Start the server
const port = process.env.PORT || 3000;
console.log(`Server is running on port ${port}`);
serve({
  fetch: app.fetch,
  port,
});

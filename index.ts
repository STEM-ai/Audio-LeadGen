import { Hono } from "hono";
import { logger } from "hono/logger";
import twilio from "twilio";
import { google } from "googleapis";
import axios from "axios";
import { WebSocketServer } from "ws";  // Corrected WebSocket import
import { fileURLToPath } from "url";
import { dirname } from "path";
import OpenAI from 'openai';
import serveStatic from 'serve-static';  
import { serve } from '@hono/node-server';

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

app.get('/', (c) => {
  return c.text('Hello, Hono on Node.js!');
});


// Create a WebSocket server
const wss = new WebSocketServer({ noServer: true });

// Handle WebSocket connections on the /stream-audio endpoint
wss.on('connection', function connection(ws) {
  console.log('WebSocket connection established');

  ws.on('message', function incoming(message) {
    console.log('Received:', message);
  });

  ws.on('close', () => {
    console.log('WebSocket connection closed');
  });
});

// Handle upgrade requests to establish WebSocket connections
app.get('/stream-audio', (c) => {
  const { req } = c;

  const server = req.raw.server;
  if (server) {
    server.on('upgrade', (request, socket, head) => {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
      });
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

// Function to handle other routes...

// Start the server
const port = process.env.PORT || 3000;
console.log(`Server is running on port ${port}`);
serve({
  fetch: app.fetch,
  port,
});

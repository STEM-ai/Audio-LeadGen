import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { logger } from 'hono/logger';
import twilio from 'twilio';
import { google } from 'googleapis';
import Sentiment from 'sentiment';
import fs from 'fs';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Google Sheets credentials setup
const GOOGLE_SHEET_ID = "10oAG2URrrX1YJr0StNAXszFycYPwBMmj9N6Y18TwcVk";
const GOOGLE_SHEET_RANGE = "Sheet1!A1:B1";

const SERVICE_ACCOUNT_INFO = {
  "type": "service_account",
  "project_id": "ferrous-thought-432910-d3",
  "private_key_id": "182b04aace7cf843455d7097555028be3a55566f",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDZX9pIDORMMchf...",
  "client_email": "leadgen@ferrous-thought-432910-d3.iam.gserviceaccount.com",
  "client_id": "116199920320342924096",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/leadgen%40ferrous-thought-432910-d3.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
};

// Set up Google Sheets API client
const auth = new google.auth.GoogleAuth({
  credentials: SERVICE_ACCOUNT_INFO,
  scopes: ['https://www.googleapis.com/auth/spreadsheets'],
});
const sheets = google.sheets({ version: 'v4', auth });

// Function to analyze negativity
const analyzeNegativity = (text: string): boolean => {
  const sentiment = new Sentiment();
  const result = sentiment.analyze(text);
  return result.score > 0; // Returns true if the sentiment is positive
};

// Function to send data to Google Sheets
const sendToGoogleSheet = async (email: string, notes: string): Promise<boolean> => {
  console.log("Triggered sendToGoogleSheet function.");
  console.log(`Data to send: Email: ${email}, Notes: ${notes}`);

  const values = [[email, notes]];
  const resource = { values };

  try {
    const result = await sheets.spreadsheets.values.append({
      spreadsheetId: GOOGLE_SHEET_ID,
      range: GOOGLE_SHEET_RANGE,
      valueInputOption: 'RAW',
      resource,
    });

    if (result) {
      console.log("Data successfully sent to Google Sheets.");
      return true;
    } else {
      console.error("Failed to send data to Google Sheets.");
      return false;
    }
  } catch (error) {
    console.error(`Exception occurred while sending data to Google Sheets: ${error}`);
    return false;
  }
};

// Function to convert text to speech using OpenAI's TTS API
const textToSpeech = async (text: string): Promise<string> => {
  const response = await openai.createAudio({
    model: 'tts-1',
    voice: 'echo',
    input: text,
  });

  const audioData = response.data.audio;
  const audioPath = '/tmp/output.mp3';

  // Convert base64 audio to binary and save as MP3 file
  const buffer = Buffer.from(audioData, 'base64');
  fs.writeFileSync(audioPath, buffer);

  return audioPath; // Return the path to the saved audio file
};

// Track user sessions and interaction count
const userSessions: { [key: string]: { exchangeCount: number; infoCollected: boolean; conversationHistory: Array<{ role: string, content: string }> } } = {};

// Define persona prompts
const initialPersonaPrompt = "You are a friendly representative of the company Coop, knowledgeable about solar energy. Your goal is to engage in a natural conversation, and answer based on the Solar Guide any questions the user may have. If a question cannot be answered by the content of the Solar Guide, answer following the industry's standards. Do not ask for personal information at this stage.";

const salesmanPersonaPrompt = "You are a friendly and persuasive solar energy salesman working for the company Coop. Your goal is to engage in a natural conversation with the user and subtly gather their email address and any specific needs or questions they may have about solar energy and Coop's products.";

// Initialize Hono app
const app = new Hono();
app.use('*', logger());

// Health check route for verifying server status
app.get('/', (c) => {
  return c.text('Server is running');
});

// Function to determine which persona to use based on exchange count
const getPersonaPrompt = (exchangeCount: number): string => {
  return exchangeCount < 3 ? initialPersonaPrompt : salesmanPersonaPrompt;
};

app.post('/incoming-call', (c) => {
  console.log('Received a POST request on /incoming-call');
  const voiceResponse = new twilio.twiml.VoiceResponse();
  voiceResponse.say('Hello, how are you');
  voiceResponse.gather({
    input: ["speech"],
    speechTimeout: "auto",
    speechModel: "experimental_conversations",
    enhanced: true,
    action: '/respond',
  });
  c.header('Content-Type', 'application/xml');
  return c.body(voiceResponse.toString());
});

app.post('/respond', async (c) => {
  const formData = await c.req.formData();
  const voiceInput = formData.get("SpeechResult")?.toString()!;

  const userId = "user_demo";

  if (!userSessions[userId]) {
    userSessions[userId] = { exchangeCount: 0, infoCollected: false, conversationHistory: [] };
  }

  const userSession = userSessions[userId];
  userSession.exchangeCount += 1;

  const personaPrompt = getPersonaPrompt(userSession.exchangeCount);

  userSession.conversationHistory.push({ role: "user", content: voiceInput });

  const systemMessage = {
    role: "system",
    content: personaPrompt
  };

  const messages = [systemMessage, ...userSession.conversationHistory];

  const chatCompletion = await openai.createChatCompletion({
    model: 'gpt-4-turbo',
    messages: messages,
  });

  const assistantResponse = chatCompletion.data.choices[0].message.content;
  userSession.conversationHistory.push({ role: "assistant", content: assistantResponse });

  const audioPath = await textToSpeech(assistantResponse);

  const voiceResponse = new twilio.twiml.VoiceResponse();
  voiceResponse.play(audioPath);
  c.header("Content-Type", "application/xml");
  return c.body(voiceResponse.toString());
});

const port = process.env.PORT || 3000;
console.log(`Server is running on port ${port}`);

serve({
  fetch: app.fetch,
  port,
});
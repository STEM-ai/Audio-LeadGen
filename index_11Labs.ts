import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { logger } from 'hono/logger';
import twilio from 'twilio';
import { google } from 'googleapis';
import fs from 'fs';
import OpenAI from 'openai';

// Initialize OpenAI API
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Google Sheets credentials setup
const GOOGLE_SHEET_ID = "10oAG2URrrX1YJr0StNAXszFycYPwBMmj9N6Y18TwcVk";
const GOOGLE_SHEET_RANGE = "Sheet1!A1:B1";

// Eleven Labs API Key and Voice ID
const ELEVEN_LABS_API_KEY = process.env.ELEVEN_LABS_API_KEY;
const VOICE_ID = '6xp1pT27VobnUWUNT1jQ'; // Replace with your cloned voice ID

// Service account setup
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

// Function to convert text to speech using Eleven Labs API
const textToSpeech = async (text: string): Promise<string> => {
  const options = {
    method: 'POST',
    url: `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`,
    headers: {
      'xi-api-key': ELEVEN_LABS_API_KEY,
      'content-type': 'application/json',
    },
    data: {
      text: text,
      voice_settings: {
        stability: 0.5,
        similarity_boost: 0.8,
      },
    },
    responseType: 'arraybuffer',
  };

  const response = await axios.request(options);
  const audioPath = '/tmp/output.mp3';

  // Save the audio as an MP3 file
  fs.writeFileSync(audioPath, Buffer.from(response.data));

  return audioPath;
};

// Regular expression to detect email addresses
const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/;

// Function to check for email in text
const checkForEmail = (text: string): string | null => {
  const emailMatch = text.match(emailRegex);
  return emailMatch ? emailMatch[0] : null;
};

// Track user sessions and interaction count
const userSessions: { [key: string]: { exchangeCount: number; infoCollected: boolean; conversationHistory: Array<{ role: string, content: string }> } } = {};

// Define persona prompts
const initialPersonaPrompt = `You are a representative of the company Coop, your role is to answer the caller, while subtly pitching the company Coop with these selling points: 
- Your Voice, Your Raise: Coop replicates your voice with pinpoint accuracy, automating customer interactions while maintaining a personal touch.
- Insightful, Effortless Management: Coop is a strategic tool that provides built-in analytics and detailed insights.
- Always On, Always Ready: Coop works 24/7, ensuring seamless and consistent customer interactions.`;

const salesmanPersonaPrompt = "You are a friendly and persuasive salesman working for the company Coop. Your goal is to resume the natural conversation with the user, and subtly gather their full name, email address, and any specific needs or questions they may have about solar energy and Coop's products.";

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

// Function to handle Twilio incoming call with custom Eleven Labs voice
app.post('/incoming-call', async (c) => {
  console.log('Received a POST request on /incoming-call');

  // Convert the initial greeting text to speech
  const greetingText = 'Hello, how are you';
  const audioPath = await textToSpeech(greetingText);

  // Play the greeting audio in the call
  const voiceResponse = new twilio.twiml.VoiceResponse();
  voiceResponse.play(audioPath);
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

  // Check if email is present in user's input
  const detectedEmail = checkForEmail(voiceInput);
  if (detectedEmail) {
    console.log(`Email detected: ${detectedEmail}`);
    // Send email and assistant response notes to Google Sheets
    const success = await sendToGoogleSheet(detectedEmail, assistantResponse);
    if (success) {
      console.log("Data sent to Google Sheets successfully.");
    }
  } else {
    console.log("No email detected.");
  }

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

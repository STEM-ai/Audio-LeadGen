import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { logger } from 'hono/logger';
import twilio from 'twilio';
import { google } from 'googleapis';
import fs from 'fs';
import OpenAI from 'openai'; // Corrected import for OpenAI
import axios, { AxiosRequestConfig } from 'axios';
import serveStatic from 'serve-static';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Fix for __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Initialize OpenAI API
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
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
  const options: AxiosRequestConfig = {
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
    responseType: 'arraybuffer' as const,
  };

  const response = await axios.request(options);
  const audioPath = './static/output.mp3';

  // Save the audio as an MP3 file
  fs.writeFileSync(audioPath, Buffer.from(response.data));
 
  // Ensure the file exists before proceeding
  try {
    if (fs.existsSync(audioPath)) {
      console.log(`Audio file successfully created at ${audioPath}`);
    } else {
      throw new Error('Audio file creation failed.');
    }
  } catch (error) {
    console.error(`Failed to confirm file creation at ${audioPath}:`, error);
    throw new Error('Audio file creation failed.');
  }

  return audioPath;
};

// Track user sessions and interaction count
const userSessions: { [key: string]: { exchangeCount: number; infoCollected: boolean; conversationHistory: Array<{ role: string, content: string }> } } = {};

// Define persona prompts
const initialPersonaPrompt = `You are a representative of the company Coop, your role is to answer the caller with very short answers, while subtly pitching the company Coop with these selling points: 
- Your Voice, Your Raise: Coop replicates your voice with pinpoint accuracy, automating customer interactions while maintaining a personal touch.
- Insightful, Effortless Management: Coop is a strategic tool that provides built-in analytics and detailed insights.
- Always On, Always Ready: Coop works 24/7, ensuring seamless and consistent customer interactions.`;

const salesmanPersonaPrompt = "You are a friendly and persuasive salesman working for the company Coop. Your goal is to resume the natural conversation with the user with very short answers, and subtly gather their full name, email address, and any specific needs or questions they may have about Coop's products.";

// Initialize Hono app
const app = new Hono();
app.use('*', logger());

// Ensure ./static directory exists
if (!fs.existsSync('./static')) {
  fs.mkdirSync('./static');
}

// Health check route for verifying server status
app.get('/', (c) => {
  return c.text('Server is running');
});

app.use('/static', serveStatic(join(__dirname, 'static')));

// Function to determine which persona to use based on exchange count
const getPersonaPrompt = (exchangeCount: number): string => {
  return exchangeCount < 3 ? initialPersonaPrompt : salesmanPersonaPrompt;
};

// Function to handle Twilio incoming call with custom Eleven Labs voice
app.post('/incoming-call', (c) => {
  console.log('Received a POST request on /incoming-call');
  const voiceResponse = new twilio.twiml.VoiceResponse();
  voiceResponse.say('Hello, how are you');
  voiceResponse.gather({
    input: ["speech"],
    speechTimeout: "auto",
    speechModel: "phone_call",
    enhanced: true,
    action: '/respond',
  });
  c.header('Content-Type', 'application/xml');
  return c.body(voiceResponse.toString());
});

app.post('/respond', async (c) => {
  try {
    // Fetch form data and handle the request
    const formData = await c.req.formData();
    const voiceInput = formData.get("SpeechResult")?.toString()!;
    const userId = "user_demo";

    if (!userSessions[userId]) {
      userSessions[userId] = { exchangeCount: 0, infoCollected: false, conversationHistory: [] };
    }

    const userSession = userSessions[userId];
    userSession.exchangeCount += 1;

    // Start background processing (OpenAI, TTS)
    processResponse(voiceInput, userSession, c);

    // Return TwiML response to pause the call while processing happens in the background
    const voiceResponse = new twilio.twiml.VoiceResponse();

    // Pause for a certain duration (e.g., 10 seconds)
    voiceResponse.pause({ length: 10 });

    // Redirect to check status to keep the call alive after the pause
    voiceResponse.redirect('/check-status');

    c.header("Content-Type", "application/xml");
    return c.body(voiceResponse.toString());
  } catch (error) {
    console.error('Error in /respond route:', error);
    return c.text('Server error', 500);
  }
});

// Check status route to continue or play the audio
app.post('/check-status', async (c) => {
  const userId = "user_demo";  // Assuming this is the demo user session
  const userSession = userSessions[userId];

  const voiceResponse = new twilio.twiml.VoiceResponse();

  if (userSession && userSession.audioReady) {
    // Use your static Replit URL to serve the audio file
    const publicAudioUrl = 'https://ecc7f349-519e-4bdd-9cf9-e6f5697e81bc-00-2w0vh2l26wc9.spock.replit.dev/static/output.mp3';

    // Tell Twilio to play the audio file
    voiceResponse.play(publicAudioUrl);
  } else {
    // If audio is not ready, keep the call alive by redirecting with a pause
    voiceResponse.pause({ length: 5 });
    voiceResponse.redirect('/check-status');
  }

  c.header("Content-Type", "application/xml");
  return c.body(voiceResponse.toString());
});

// Function to handle the background processing (unchanged from previous version)
const processResponse = async (voiceInput: string, userSession: any, c: any) => {
  try {
    const personaPrompt = getPersonaPrompt(userSession.exchangeCount);

    // Add user input to conversation history
    userSession.conversationHistory.push({ role: "user", content: voiceInput });

    // Construct system message
    const systemMessage = {
      role: "system",
      content: personaPrompt,
    };

    // Combine system message with conversation history
    const messages = [systemMessage, ...userSession.conversationHistory];

    // Fetch response from OpenAI
    const chatCompletion = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: messages,
    });

    // Log the OpenAI response
    console.log("OpenAI Response:", chatCompletion);

    // Extract assistant's response
    const assistantResponse = chatCompletion.choices[0].message.content;
    userSession.conversationHistory.push({ role: "assistant", content: assistantResponse });

    // Check if email is present in user's input and send to Google Sheets
    const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/;
    const detectedEmail = voiceInput.match(emailRegex)?.[0];
    if (detectedEmail) {
      console.log(`Email detected: ${detectedEmail}`);
      const success = await sendToGoogleSheet(detectedEmail, assistantResponse);
      if (success) {
        console.log("Data sent to Google Sheets successfully.");
      }
    }

    // Convert the assistant's response to speech and save audio file
    const audioPath = await textToSpeech(assistantResponse);
    console.log("Output from ElevenLabs received");
    // Serve the generated audio file via Twilio
    const twilioResponse = new twilio.twiml.VoiceResponse();
    twilioResponse.play(audioPath);
    c.header("Content-Type", "application/xml");
    console.log("Answer played");

  } catch (error) {
    console.error('Error in processResponse function:', error);
  }
};

app.get('/static/output.mp3', (c) => {
  const filePath = join(__dirname, 'static', 'output.mp3');
  if (fs.existsSync(filePath)) {
    return c.file(filePath);
  } else {
    return c.text('File not found', 404);
  }
});

const port = process.env.PORT || 3000;
console.log(`Server is running on port ${port}`);

serve({
  fetch: app.fetch,
  port,
});

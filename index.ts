import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { logger } from 'hono/logger';
import twilio from 'twilio';
import { google } from 'googleapis';
import fs from 'fs';
import OpenAI from 'openai';
import axios from 'axios';
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
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDZX9pIDORMMchf\nQFE/cAV4VPVEpaq8rW0GJSIPnF2dXRoegSy7O5YNStoSWgMu74PDUreO2CGi/Wku\nlh8pbUGBsPbf4VkLNqmVGcqL6RkG5d4AhZQp5RomX+WyOwIhpd65SrqHXtRIpKyW\nteSrei5+G+mjYWNtH6lVoEkF7ezdXd2D/gFa9Q65GHJKF9E6hInRph346rvbTd8Z\nkB38/TOOc6GBr3XTt1x4dY5nBJqsX4Bqe2nxDvyveQJ4v0rPrYStZRM4KlnFKTWP\nkexwXwfnWxtNNeC1TVIDvRWiVV0JS99ymyODmJn33eLmziFN0EJJdquD0iTfRf51\ncBsH8UzNAgMBAAECggEACIoF6qeq/j1EaE2AA5R0eo4n1msFooTTkBa8WE2ltc1W\n/dTIO5CzK9GBcJAdqOXa0Lz6nf9qjtsSmzRlg/yZQq1/fTr+gvzCO6u4M7fT9lvo\nVS/qKp0n4lMJFG/R/R1levTvD+tPPo1NhFwf4AacNfMFwhfMzpgcUFNMGxIGKIn5\nVa4rJifeuhCJNK8HsyAI3ChnvXcqvQE94nmIpKv4va+tsHB9vg+4c/mcJZ+V7U9d\n/a/MUFzf7TkQtsHaz6HN+wHR4QHg8bohEt+4IyHWV2BPbAJ8J0SgWzXMyDS/54Pk\n2cnoDCp0bliR1D6X7YlCAQrtpJnGyLJ/ZU1S4EaVcQKBgQD//qRFaBZp2CmkyoMi\nIfjcBmEacL0Z3CUMAMEowgxmUBiPme/vePVKM2ZfK4dBTtmTbGmKZzZ2iCZy5psF\nMnmMc18XfVoqAaNEnYHnvVYj/kv+hJ8wYkyyj64U7qALXp4SBvZ+BAY9qgj5/E6t\nuFGBRYV9j9kmVkzhoXWF8nU++QKBgQDZYQGM+3gWMYANkENBv+yU3bBQ9NJ6jY4P\n7kHx4PBc6f3cP1WB0y4CMPeBpGwKxVsJ350I8BkHnE+Ixu+QStub3lzNjvxXYPlb\nfeQqZpIEKAmDqzbdfjrH8LyewZOZVZTB95MpcZnCS9fGrbMw8sxPhzSURUUsoGmE\nQyCZZkftdQKBgQDl9OGts2XG6LXn4T7Qz4GUbGqX7MQB0d65nIfnTAEFe1fEz3xY\nOujlIa0JOrnCMcmDA7T+7d5ftcgMGRkSHxhO0WiPWjw/Vb9LKM4D1PHnXUz4sjup\no/PPxv+SsBS2geUuvnB4HLdadz6fCUXICbW1kTTr6Ocg6A8h8/71NyqZSQKBgQCx\nlF+R7nSBnNqBOhLXiZQZYKkC2Z2AZFdjiD3y/NEe9kBeRpbxwbTaMWpgTBO/EM54\nWGaOwKWR5A3NLMbT13Nj99lUS7S1JRFPvp5ATR6HqrVrDNl7Q/19DJrqDjUnlBQ8\nCKX9u0HiydZyBcBXAmIJreg0IAqMlFbep3/gEQA9aQKBgGlrZ7W1G5tdmCdsEZ/m\nJXFyHaPKg8k/mP5R+GkRaIdjpZ/nE+95pwy8Mp3KUFbcloI//y5l1tmxYRxX2JbN\nd6I1hfCRdCUFNuKEMRiYM0ClrMt5fXUFVoibWD6vuYwLiFn/iZe006XRxHrw2zK1\nXwyfGKc/ObC7K3vt2rBnb1dQ\n-----END PRIVATE KEY-----\n",
  "client_email": "leadgen@ferrous-thought-432910-d3.iam.gserviceaccount.com",
  "client_id": "116199920320342924096",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/leadgen%40ferrous-thought-432910-d3.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
};

// Authenticate with Google using the service account info
const auth = new google.auth.GoogleAuth({
  credentials: SERVICE_ACCOUNT_INFO,
  scopes: ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
});

const sheets = google.sheets({ version: 'v4', auth });
const drive = google.drive({ version: 'v3', auth });

// Function to upload file to Google Drive
const uploadToGoogleDrive = async (filePath, fileName) => {
  const fileMetadata = {
    name: fileName,
    parents: ['1VY5oBAgAM54lLh2lfMGk55vEWoT_f1zC'],  // Replace with your Google Drive folder ID
  };
  const media = {
    mimeType: 'audio/mp3',
    body: fs.createReadStream(filePath),
  };

  const response = await drive.files.create({
    resource: fileMetadata,
    media: media,
    fields: 'id',
  });

  const fileId = response.data.id;

  // Set file permissions to public
  await drive.permissions.create({
    fileId: fileId,
    requestBody: {
      role: 'reader',
      type: 'anyone',
    },
  });

  // Get public link
  const result = await drive.files.get({
    fileId: fileId,
    fields: 'webViewLink, webContentLink',
  });

  return result.data.webContentLink;
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

// Function to convert text to speech using Eleven Labs and upload to Google Drive
const textToSpeechAndUpload = async (text: string) => {
  try {
    // Create an Eleven Labs API request for text-to-speech
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

    // Await the API request to ensure it's completed
    const response = await axios.request(options);

    // Check if the response is valid before processing
    if (!response || !response.data) {
      throw new Error('Invalid response from Eleven Labs API');
    }

    console.log("Eleven Labs response:", response);
    console.log("Eleven Labs response headers:", response.headers);  // Log headers for debugging

    const audioPath = './static/output.mp3';

    // Write the audio data to the file
    fs.writeFileSync(audioPath, Buffer.from(response.data));

    // Upload to Google Drive and get public link
    console.log("Uploading audio to Google Drive...");
    const publicAudioUrl = await uploadToGoogleDrive(audioPath, 'output.mp3');
    console.log("Public URL from Google Drive:", publicAudioUrl);

    // Check if the publicAudioUrl is valid
    if (!publicAudioUrl) {
      throw new Error('Failed to upload audio to Google Drive');
    }

    return publicAudioUrl;
  } catch (error) {
    console.error('Error in textToSpeechAndUpload function:', error.message || error);
    throw error;
  }
};

// Track user sessions and interaction count
const userSessions: { [key: string]: { exchangeCount: number; infoCollected: boolean; conversationHistory: Array<{ role: string, content: string }>, publicAudioUrl?: string } } = {};

// Define persona prompts
const initialPersonaPrompt = `You are a representative of the company Coop, your role is to answer the caller with concise answers, while subtly pitching the company Coop with these selling points: 
- Your Voice, Your Raise: Coop replicates your voice with pinpoint accuracy, automating customer interactions while maintaining a personal touch.
- Insightful, Effortless Management: Coop is a strategic tool that provides built-in analytics and detailed insights.
- Always On, Always Ready: Coop works 24/7, ensuring seamless and consistent customer interactions.`;

const salesmanPersonaPrompt = `You are a friendly and persuasive salesman working for the company Coop. Your goal is to resume the natural conversation with the user with concise answers, and subtly gather their full name, email address, and any specific needs or questions they may have about Coop's products.`;

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
    speechTimeout: "5", // "auto"
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
    const publicAudioUrl = await processResponse(voiceInput, userSession);

    // Serve the generated audio file via Twilio and gather new input
    const twilioResponse = new twilio.twiml.VoiceResponse();
    twilioResponse.play(publicAudioUrl);

    // Gather more input after the response is played
    twilioResponse.gather({
      input: ["speech"],
      speechTimeout: "5", // "auto"
      speechModel: "phone_call",
      enhanced: true,
      action: '/respond', // Loop back to the same /respond route
    });

    c.header("Content-Type", "application/xml");
    return c.body(twilioResponse.toString());
  } catch (error) {
    console.error('Error in /respond route:', error);
    return c.text('Server error', 500);
  }
});


// Function to handle the background processing
const processResponse = async (voiceInput: string, userSession: any) => {
  try {
    const personaPrompt = getPersonaPrompt(userSession.exchangeCount);
    console.log(`Persona prompt: ${personaPrompt}`);
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

    // Log the OpenAI response
    console.log("OpenAI Response:", assistantResponse);

    // Convert the assistant's response to speech and get the public audio URL
    const publicAudioUrl = await textToSpeechAndUpload(assistantResponse);
    console.log("Output from ElevenLabs received");

    // Store the public audio URL in user session for future use if needed
    userSession.publicAudioUrl = publicAudioUrl;

    return publicAudioUrl;
  } catch (error) {
    console.error('Error in processResponse function:', error);
    throw error;
  }
};

const port = process.env.PORT || 3000;
console.log(`Server is running on port ${port}`);

serve({
  fetch: app.fetch,
  port,
});
import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { logger } from 'hono/logger';
import twilio from 'twilio'; 
import OpenAI from 'openai';
import { google } from 'googleapis';
import Sentiment from 'sentiment';

// Retrieve the API key from environment variables
const openaiApiKey = process.env.OPENAI_API_KEY;
if (!openaiApiKey) {
  throw new Error('OpenAI API key not found in environment variables');
}
const openai = new OpenAI({
  apiKey: openaiApiKey,
});

// Google Sheets credentials setup
const GOOGLE_SHEET_ID = "10oAG2URrrX1YJr0StNAXszFycYPwBMmj9N6Y18TwcVk"; // Your Google Sheet ID
const GOOGLE_SHEET_RANGE = "Sheet1!A1:B1"; // Adjust range to include only email and notes

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

// Set up Google Sheets API client
const auth = new google.auth.GoogleAuth({
  credentials: SERVICE_ACCOUNT_INFO,
  scopes: ['https://www.googleapis.com/auth/spreadsheets'],
});
const sheets = google.sheets({ version: 'v4', auth });

// Function to analyze negativity
const analyzeNegativity = (text) => {
  const sentiment = new Sentiment();
  const result = sentiment.analyze(text);
  return result.score > 0; // Returns true if the sentiment is positive
};

// Function to send data to Google Sheets
const sendToGoogleSheet = async (email, notes) => {
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

// Function to extract information from input text
const analyzeInputForInformation = (inputText) => {
  let email = "N/A";
  let notes = "N/A";
  let detect = false;

  // Regular expressions for detecting emails
  const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;

  // Extract email if present
  const emailMatch = inputText.match(emailRegex);
  if (emailMatch) {
    email = emailMatch[0];
    detect = true;
  }

  // For notes, you might want a more sophisticated NLP extraction
  notes = inputText.replace(email, "").trim(); // Basic example

  return { email, notes, detect };
};

// Track user sessions and interaction count
const userSessions = {}; // { userId: { exchangeCount: 0, infoCollected: false, conversationHistory: [] } }

// Define persona prompts
const initialPersonaPrompt = `You are a representative of the company Coop, your role is to answer the caller with briefs response, while subtly pitching the company Coop with these selling points in just a few sentences: 
- Your Voice, Your Raise: Coop replicates your voice with pinpoint accuracy, automating customer interactions while maintaining a personal touch.
- Insightful, Effortless Management: Coop is a strategic tool that provides built-in analytics and detailed insights.
- Always On, Always Ready: Coop works 24/7, ensuring seamless and consistent customer interactions.`;

const salesmanPersonaPrompt = "You are a friendly and persuasive salesman working for the company Coop. Your goal is to resume the natural conversation with the user, and subtly gather their email address, and any specific needs or questions they may have about Coop's products, in a concise  way.";

// Initialize Hono app
const app = new Hono();
app.use('*', logger());

// Health check route for verifying server status
app.get('/', (c) => {
  return c.text('Server is running');
});

// Function to determine which persona to use based on exchange count
const getPersonaPrompt = (exchangeCount) => {
  if (exchangeCount < 3) {
    return initialPersonaPrompt;
  } else {
    return salesmanPersonaPrompt;
  }
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

  // Dummy user_id for example, replace with actual logic
  const userId = "user_demo";

  if (!userSessions[userId]) {
    userSessions[userId] = { exchangeCount: 0, infoCollected: false, conversationHistory: [] };
  }

  const userSession = userSessions[userId];
  userSession.exchangeCount += 1;

  // Determine the persona prompt based on the exchange count
  const personaPrompt = getPersonaPrompt(userSession.exchangeCount);

  // Add user message to conversation history
  userSession.conversationHistory.push({ role: "user", content: voiceInput });

  // Prepare the system message with persona prompt
  const systemMessage = {
    role: "system",
    content: personaPrompt
  };

  // Combine the system message and conversation history
  const messages = [systemMessage, ...userSession.conversationHistory];

  // Call the OpenAI API with more context
  const chatCompletion = await openai.chat.completions.create({
    model: 'gpt-4-turbo',
    messages: messages,
    temperature: 0,
  });

  // Add assistant response to conversation history
  const assistantResponse = chatCompletion.choices[0].message.content;
  userSession.conversationHistory.push({ role: "assistant", content: assistantResponse });

  // Analyze user input for email and notes if needed (only if salesman persona is active)
  if (userSession.exchangeCount >= 3 && !userSession.infoCollected) {
    const { email, notes, detect } = analyzeInputForInformation(voiceInput);

    if (detect) {
      console.log(`Extracted Information - Email: ${email}, Notes: ${notes}`);

      // If all information is collected and valid, send it to Google Sheets
      if (sendToGoogleSheet(email, notes)) {
        userSession.infoCollected = true;
        const voiceResponse = new twilio.twiml.VoiceResponse();
        voiceResponse.say("Thank you! Your information has been collected successfully. We'll be in touch soon.");
        c.header("Content-Type", "application/xml");
        return c.body(voiceResponse.toString());
      } else {
        console.error("Failed to send information to Google Sheets.");
      }
    }
  }

  const voiceResponse = new twilio.twiml.VoiceResponse();
  voiceResponse.say(assistantResponse);
  c.header("Content-Type", "application/xml");
  return c.body(voiceResponse.toString());
});

const port = process.env.PORT || 3000; // Ensure PORT is configurable via env variable.
console.log(`Server is running on port ${port}`);

serve({
  fetch: app.fetch,
  port,
});

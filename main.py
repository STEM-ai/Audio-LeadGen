import os
import time
import json
from fastapi import FastAPI, Request, Response
import uvicorn
import requests
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import asyncio
import logging
from textblob import TextBlob
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from twilio.twiml.voice_response import VoiceResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
#from cartesia import Cartesia

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the server time to ensure clock synchronization
logger.info(f"Server time at startup: {time.ctime()}")

client = OpenAI()

app = FastAPI()

# Mount the static directory for serving static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define a Pydantic model for Twilio requests
class TwilioRequest(BaseModel):
    To: str = None
    From: str = None
    CallSid: str = None
    RecordingUrl: str = None
    RecordingSid: str = None

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Define retriever as a global variable
retriever = None

AIRTABLE_PERSONAL_TOKEN = os.getenv('AIRTABLE_PERSONAL_TOKEN')
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
AIRTABLE_TABLE_NAME = "LeadGenapp"

GOOGLE_SHEET_ID = "10oAG2URrrX1YJr0StNAXszFycYPwBMmj9N6Y18TwcVk"
GOOGLE_SHEET_RANGE = "Sheet1!A1:C1"

service_account_info = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

# Parse the JSON from the environment variable
if service_account_info:
    service_account_info = json.loads(service_account_info)

    # Create the service account credentials
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"])
else:
    raise EnvironmentError(
        "Service account info is missing in the environment variables")

service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

#logger.info(f"GOOGLE_SHEET_ID: {GOOGLE_SHEET_ID}, GOOGLE_SHEET_RANGE: {GOOGLE_SHEET_RANGE}")

def generate_voice_response(text: str) -> str:
    # Define the path to save the audio file
    speech_file_path = os.path.join(os.path.dirname(__file__),
                                    "static/speech.mp3")

    # client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))

    # data = client.tts.bytes(
    #     model_id="sonic-english",
    #     transcript=text,
    #     voice_id="829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30",  
    #     output_format={
    #         "container": "mp3",
    #         "encoding": "pcm_f32le",
    #         "sample_rate": 44100,
    #     },
    # )

    # with open(speech_file_path, "wb") as f:
    #     f.write(data)
        
    #OpenAI
    
    response = client.audio.speech.create(model="tts-1",
                                          voice="nova",
                                          input=text)
    response.stream_to_file(speech_file_path)

    audio_url = f"{os.getenv('BASE_URL', 'https://be9b7102-f7ca-4519-bba7-9a93be845fb9-00-qynkptyk0lr7.riker.replit.dev')}/get-audio"

    return audio_url


def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text")
    return transcription

def download_recording_with_retry(recording_url, account_sid, auth_token, max_retries=5, delay=2):
    for attempt in range(max_retries):
        response = requests.get(recording_url, auth=(account_sid, auth_token))

        if response.status_code == 200:
            # Save the recording to a file
            audio_file_path = "temp_recording.wav"
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(response.content)
            logger.info(f"Recording downloaded successfully after {attempt + 1} attempt(s).")
            return audio_file_path

        logger.info(f"Attempt {attempt + 1} failed. Recording not yet available. Status code: {response.status_code}")

        # Wait before retrying
        time.sleep(delay)

    # If the recording is still not available after all retries, return None
    logger.error("Recording not available after maximum retries.")
    return None

def analyze_negativity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity > 0  # Positive polarity indicates valid information


def send_to_airtable(fullname, email, phone, notes):
    #logger.info("Triggered send_to_airtable function.")
    #logger.info(f"Data to send: Full Name: {fullname}, Email: {email}, Phone: {phone}, Notes: {notes}")

    airtable_url = "https://api.airtable.com/v0/appUFMzXqi8COgNVK/tblrvpXEJwY0fmeJR"
    #f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

    # Log the URL being used for the Airtable API call
    #logger.info(f"Using Airtable URL: {airtable_url}")

    headers = {
        "Authorization":
        f"Bearer pattTKnZnWMKylfKB.c1a91c114dc862bbc68ef761bf580fa50fd0efe76d3a1cc8f9a1495dd3f4954b",  #{AIRTABLE_PERSONAL_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "fields": {
            "Full Name": fullname,
            "Email": email,
            "Phone": phone,
            "Notes": notes
        }
    }

    # Log the data payload
    #logger.info(f"Payload being sent to Airtable: {data}")

    try:
        response = requests.post(airtable_url, headers=headers, json=data)

        # Log the status code and response text
        #logger.info(f"Response Status Code: {response.status_code}")

        if response.status_code == 200 or response.status_code == 201:
            #logger.info("Data successfully sent to Airtable.")
            return True
        else:
            #logger.error(f"Failed to send data to Airtable. Status code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        #logger.error(f"Exception occurred while sending data to Airtable: {e}")
        return False


def send_to_google_sheet(fullname, email, phone, notes):

    #logger.info(f"Data to send: Full Name: {fullname}, Email: {email}, Phone: {phone}, Notes: {notes}")

    values = [[fullname, email, phone, notes]]
    body = {'values': values}

    try:
        result = sheet.values().append(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=
            GOOGLE_SHEET_RANGE,  # Adjust the range to include the phone column
            valueInputOption="RAW",
            body=body).execute()

        if result:
            #logger.info("Data successfully sent to Google Sheets.")
            return True
        else:
            logger.error("Failed to send data to Google Sheets.")
            return False
    except Exception as e:
        logger.error(
            f"Exception occurred while sending data to Google Sheets: {e}")
        return False


DOCUMENT_PATH1 = "docs/HR_internal.pdf"
DOCUMENT_PATH2 = "docs/Trainee_template.pdf"
HASH_FILE = "document_hash.txt"
VECTOR_STORE_PATH = "faiss_vector_db"

# Initialize Conversation Buffer Memory
conversation_memory = ConversationBufferMemory(return_messages=True)


def document_changed():
    document_mtime = os.path.getmtime(
        DOCUMENT_PATH1)  # Get the document's last modified time

    if os.path.exists(
            HASH_FILE
    ):  # Reuse the HASH_FILE for storing the modification time
        with open(HASH_FILE, 'r') as f:
            stored_mtime = f.read()

        # If the modification time is the same, no need to reprocess the document
        if str(document_mtime) == stored_mtime:
            return False

    # Document has changed, update the stored modification time
    with open(HASH_FILE, 'w') as f:
        f.write(str(document_mtime))

    return True


def ingest_docs():
    loader1 = UnstructuredPDFLoader(DOCUMENT_PATH1)
    docs1 = loader1.load()
    loader2 = UnstructuredPDFLoader(DOCUMENT_PATH2)
    docs2 = loader2.load()
    docs = docs1 + docs2
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()

    if not chunks:
        raise ValueError(
            "No chunks created from the document. Check the document loading process."
        )

    # Create or overwrite the vector store
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)

    return vectorstore  # Return the vectorstore to update retriever globally


if document_changed():
    conversation_memory.clear()
    conversation_memory = ConversationBufferMemory(return_messages=True)
    vectorstore = ingest_docs()
    retriever = vectorstore.as_retriever()
else:
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH,
                                   OpenAIEmbeddings(),
                                   allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

# Initialize the Conversation Chain with memory
conversation_chain = ConversationChain(
    llm=llm,
    memory=conversation_memory,
)

app = FastAPI(
    title="LangChain Server with Knowledge Base",
    version="1.0",
    description=
    "A knowledge-based API server using LangChain and Solar Guide as a knowledge source",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the root endpoint!"}


initial_persona_prompt = (
    "You are a friendly representative of Kay Soley, knowledgeable about solar energy. Answer in the same language as the user. Your goal is to engage in a natural conversation, and answer based on the Solar Guide any questions the user may have. Do not ask for contact information at this stage.\n If a question cannot be answered by the content of the Solar Guide, say that you are unsure and that the user should ask this question to one of our Technicians during a telephone or home appointment.\n At the end of each answer, ask them open-ended questions to learn more about their project. Always refer to the conversation history. Do not repeat questions that have already been asked. No need to greet yourself if already done in the conversation. Clarity and Conciseness: Use bullet points or numbered lists for clarity in your responses, and keep responses concise, limited to 2-3 sentences."
)

salesman_persona_prompt = (
    "You are a friendly and persuasive solar energy salesman working for Kay Soley. Answer in the same language as the user. "
    "Always refer to the conversation history. Do not repeat questions that have already been asked. No need to greet yourself if already done in the conversation.Your goal is to resume a natural conversation with the user, subtly gather their full name, email address, phone number, and any specific needs or questions they may have about solar energy and Kay Soley with brief answers. "
    "Clarity and Conciseness: Use bullet points or numbered lists for clarity in your responses, and keep responses concise, limited to 2-3 sentences.\n"
    "- Use a polite and non-intrusive approach when asking for the user's full name, phone number and email address.\n"
    "- Ensure the conversation feels natural and the user feels comfortable providing their information.\n\n"
    "Desired Format:\n"
    "- Gather the following information from the user:\n"
    "  - Full Name: <extracted full name if any>\n"
    "  - Email: <extracted email address if any>\n"
    "  - Phone: <extracted phone number if any>\n"
    "  - Notes: <Brief and precise summary of the user's needs or questions>")

faq = (
    "You are a friendly representative of Kay Soley, knowledgeable about solar energy. Answer in the same language as the user. Your goal is to answer questions the user may have. Do not ask for contact information at this stage. Don't continue asking questions to the user.\n If a question cannot be answered by the content of the Solar Guide, say that you are unsure and that the user should ask this question to one of our Technicians during a telephone or home appointment.\n Always refer to the conversation history. Do not repeat questions that have already been asked. No need to greet yourself if already done in the conversation. Clarity and Conciseness: Use bullet points or numbered lists for clarity in your responses, and keep responses concise, limited to 2-3 sentences."
)

# Dictionaries to track user state
exchange_count = {}
user_data = {}
info_collected = {}
salesman_prompt_given = {}
conversation_memory_dict = {}
last_active_time = {}


def analyze_input_for_information(input_text, conversation_history, user_id):
    #logger.info(f"Conversation history for user {user_id}: {conversation_history}")
    """Analyzes the user's input and full conversation history to check for personal information and create a summary."""

    verification_prompt = f"""
    You are to analyze the following conversation and input to verify if it contains user profile and project, and provide a brief summary of their needs, intent, or questions based on the entire 'Conversation'.

    Conversation:
    {conversation_history}

    Latest Input:
    {input_text}

    Please respond with the information found in the following format:
    Full Name: <extracted full name if any>
    Email: <extracted email address if any>
    Phone: <extracted phone number if any>
    Notes: <Brief summary of the user's questions and profile based on the entire conversation>
    """
    verification_result = llm.invoke(verification_prompt)

    extracted_data = verification_result.content.strip().split("\n")

    fullname = user_data.get(user_id, {}).get('fullname', 'N/A')
    email = user_data.get(user_id, {}).get('email', 'N/A')
    phone = user_data.get(user_id, {}).get('phone', 'N/A')
    notes = "N/A"
    detect = False

    for item in extracted_data:
        if item.startswith("Full Name:"):
            extracted_fullname = item.replace("Full Name:", "").strip()
            if extracted_fullname and extracted_fullname.lower(
            ) != "not provided":
                fullname = extracted_fullname
        elif item.startswith("Email:"):
            extracted_email = item.replace("Email:", "").strip()
            if extracted_email and "@" in extracted_email:
                email = extracted_email
        elif item.startswith("Phone:"):
            extracted_phone = item.replace("Phone:", "").strip()
            if extracted_phone and len(extracted_phone) >= 10:
                phone = extracted_phone
        elif item.startswith("Notes:"):
            extracted_notes = item.replace("Notes:", "").strip()
            if extracted_notes and extracted_notes.lower() != "not provided":
                notes = extracted_notes

    user_data[user_id] = {
        "fullname": fullname,
        "email": email,
        "phone": phone,
        "notes": notes
    }

    detect = all(value != "N/A" for value in user_data[user_id].values())

    return fullname, email, phone, notes, detect


def clear_cache():
    cache_path = ".cache/*"
    try:
        os.system(f"rm -rf {cache_path}")
        #logger.info(".cache directory has been cleared.")
    except Exception as e:
        logger.error(f"Error while clearing cache: {e}")


def cleanup_sessions(timeout=3600 * 6):  # Timeout in seconds
    current_time = time.time()
    to_remove = []
    for session_id, last_active in last_active_time.items():
        if current_time - last_active > timeout:
            to_remove.append(session_id)

    for session_id in to_remove:
        exchange_count.pop(session_id, None)
        user_data.pop(session_id, None)
        info_collected.pop(session_id, None)
        salesman_prompt_given.pop(session_id, None)
        conversation_memory_dict.pop(session_id, None)
        last_active_time.pop(session_id, None)
        #logger.info(f"Session {session_id} has been cleaned up due to inactivity.")


# Background task to clear memory and cache
async def periodic_reset():
    while True:
        await asyncio.sleep(3600 * 6)
        clear_cache()
        cleanup_sessions()


@app.on_event("startup")
async def startup_event():
    # Start the background task when the FastAPI app starts
    asyncio.create_task(periodic_reset())
    #logger.info("Started background tasks")


@app.post("/incoming-call")
async def incoming_call(request: Request):
    #form_data = await request.form()
    #twilio_request = TwilioRequest(**form_data)
    #session_id = twilio_request.CallSid

    # Set the audio URL to the pre-recorded welcome message
    audio_url = f"{os.getenv('BASE_URL', 'https://be9b7102-f7ca-4519-bba7-9a93be845fb9-00-qynkptyk0lr7.riker.replit.dev')}/play-welcome-audio"

    # Create TwiML response: play audio and record input without transcription
    response = VoiceResponse()
    response.play(audio_url)

    # Record user input without Twilio transcription
    response.record(
        action=
        "/process_recording",  # Send to process_recording to handle the input
        method="POST",
        max_length=60,
        timeout=1,  # Ends recording after n seconds of silence
        finish_on_key="#"  # Optional: Allows caller to end input with "#"
    )

    # Redirect if no input is gathered
    response.redirect("/process_recording")

    return Response(content=str(response), media_type="application/xml")

@app.post("/process_recording")
async def process_recording(request: Request):
    try:
        form_data = await request.form()

        recording_url = form_data.get("RecordingUrl")
        session_id = form_data.get("CallSid")

        if not recording_url or not session_id:
            logger.error("Missing required fields: RecordingUrl or CallSid")
            response = VoiceResponse()
            response.say("We encountered an error processing your input.")
            response.hangup()
            return Response(content=str(response), media_type="application/xml")

        # Log the recording URL
        logger.info(f"Recording URL received: {recording_url}")

        # Use Twilio credentials from environment variables
        twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")

        # Attempt to download the recording with retries
        audio_file_path = download_recording_with_retry(recording_url, twilio_account_sid, twilio_auth_token)

        if not audio_file_path:
            response = VoiceResponse()
            response.say("Sorry, we could not access your response at this time.")
            response.hangup()
            return Response(content=str(response), media_type="application/xml")

        # Transcribe the downloaded audio file using Whisper
        input_text = transcribe_audio(audio_file_path)

        # Clean up the temporary file
        os.remove(audio_file_path)

        if not input_text:
            logger.error("Transcription failed.")
            response = VoiceResponse()
            response.say("Sorry, we could not understand your response.")
            response.hangup()
            return Response(content=str(response), media_type="application/xml")

        # Log the transcription result for debugging
        logger.info(f"Transcription: {input_text}")

        # Conversation logic (update session data, etc.)
        last_active_time[session_id] = time.time()

        # Initialize session-specific data if not present
        if session_id not in conversation_memory_dict:
            
            conversation_memory_dict[session_id] = ConversationBufferMemory()
            logger.info(
                f"New conversation memory initialized for session {session_id}"
            )

        conversation_memory = conversation_memory_dict[session_id]
        
        # Initialize the Conversation Chain with memory
        conversation_chain = ConversationChain(
            llm=llm,
            memory=conversation_memory,
        )

        # Set up user-specific data
        user_id = session_id
        if session_id not in exchange_count:
            exchange_count[session_id] = 0
        if session_id not in info_collected:
            info_collected[session_id] = False
        if session_id not in user_data:
            user_data[session_id] = {
                "fullname": "N/A",
                "email": "N/A",
                "phone": "N/A",
                "notes": "N/A"
            }
        if session_id not in salesman_prompt_given:
            salesman_prompt_given[session_id] = False

        exchange_count[user_id] += 1

        # Determine persona prompt based on conversation state
        if exchange_count[user_id] < 3:
            persona_prompt = initial_persona_prompt
        elif info_collected[user_id]:
            persona_prompt = faq
        else:
            persona_prompt = salesman_persona_prompt

        # Retrieve context and prepare conversation response
        context = retriever.get_relevant_documents(input_text)
        context_text = "\n\n".join([doc.page_content for doc in context])
        question_with_context = f"{persona_prompt}\n\nContext:\n{context_text}\n\nQuestion: {input_text}"

        # Generate conversation response
        conversation_response = conversation_chain.run(question_with_context)
        conversation_memory.save_context({"input": input_text},
                                         {"output": conversation_response})

        logger.info(f"Conversation response: {conversation_response}")
        
        # Analyze input to collect user information if in salesman persona mode
        if persona_prompt == salesman_persona_prompt:
            conversation_history = conversation_memory.buffer
            fullname, email, phone, notes, detect = analyze_input_for_information(
                input_text, conversation_history, user_id)

            # Update user data with any newly collected information
            user_data[user_id] = {
                "fullname": fullname,
                "email": email,
                "phone": phone,
                "notes": notes
            }

            # If all required information is collected, send it to Airtable
            if detect and not info_collected[
                    user_id]:  # Only send if not already sent
                if send_to_airtable(fullname, email, phone, notes):
                    logger.info(
                        f"User data successfully sent to Airtable for session {user_id}"
                    )
                    info_collected[user_id] = True  # Mark as sent to Airtable
                else:
                    logger.error("Failed to send user data to Airtable.")

        # Generate TTS audio for the response
        response_audio_url = generate_voice_response(conversation_response)

        if not response_audio_url:
            logger.error("Failed to generate TTS audio for the response.")
            response = VoiceResponse()
            response.say(
                "Sorry, we are unable to process your request at the moment.")
            response.hangup()
            return Response(content=str(response),
                            media_type="application/xml")

        # Send the response to the user and wait for additional input with silence detection
        response = VoiceResponse()
        response.play(response_audio_url)

        # Record next input with a 3-second silence timeout
        response.record(
            action="/process_recording",
            method="POST",
            max_length=60,
            timeout=3,  # Ends recording after 3 seconds of silence
            finish_on_key="#"  # Optional: Allows caller to end input with "#"
        )

        return Response(content=str(response), media_type="application/xml")

    except Exception as e:
        # Log the exception for debugging
        logger.error(f"Error in process_recording: {e}")
        response = VoiceResponse()
        response.say("An error occurred while processing your response.")
        response.hangup()
        return Response(content=str(response), media_type="application/xml")


@app.get("/get-audio")
async def get_audio():
    file_path = "static/speech.mp3"
    if not os.path.exists(file_path):
        return {"error": "Audio file not found."}

    return FileResponse(file_path, media_type="audio/mpeg")


@app.get("/play-welcome-audio")
async def play_welcome_audio():
    file_path = "static/Welcome.mp3"
    if not os.path.exists(file_path):
        return {"error": "Audio file not found."}

    return FileResponse(file_path, media_type="audio/mpeg")


if __name__ == "__main__":
    logger.info("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))

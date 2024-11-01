import os
import time
import json
from fastapi import FastAPI, Request
import uvicorn
import requests
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import asyncio
import logging
from textblob import TextBlob  # Import for sentiment analysis
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the server time to ensure clock synchronization
logger.info(f"Server time at startup: {time.ctime()}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define retriever as a global variable
retriever = None


AIRTABLE_PERSONAL_TOKEN = os.getenv(
    'AIRTABLE_PERSONAL_TOKEN')  
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
        "Authorization": f"Bearer pattTKnZnWMKylfKB.c1a91c114dc862bbc68ef761bf580fa50fd0efe76d3a1cc8f9a1495dd3f4954b",#{AIRTABLE_PERSONAL_TOKEN}",
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

DOCUMENT_PATH1 = "docs/Solar_guide.pdf"
DOCUMENT_PATH2 = "docs/Tarifs_prime_autoconsommatio_V2.pdf"
HASH_FILE = "document_hash.txt"
VECTOR_STORE_PATH = "faiss_vector_db"

# Initialize Conversation Buffer Memory
conversation_memory = ConversationBufferMemory(return_messages=True)

def document_changed():
    document_mtime = os.path.getmtime(DOCUMENT_PATH1)  # Get the document's last modified time

    if os.path.exists(HASH_FILE):  # Reuse the HASH_FILE for storing the modification time
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

    print("Document successfully ingested into knowledge base")

    return vectorstore  # Return the vectorstore to update retriever globally

if document_changed():
    print("Document changed. Ingesting new document...")
    conversation_memory.clear()
    conversation_memory = ConversationBufferMemory(return_messages=True)
    print("Conversation memory cleared")
    vectorstore = ingest_docs()
    retriever = vectorstore.as_retriever(
    )
else:
    print("Document unchanged. Loading existing vector store.")
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH,
                                   OpenAIEmbeddings(),
                                   allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(
    )

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
    "*Understanding the User's Profile* but don't ask all these questions at once:"
  "- Ask which commune they live in."
  "- Inquire about their electricity bill amount and whether it's monthly or bi-monthly."
  "Try to understand their consumption habits:"
  "- Is the house typically occupied during the day or mainly in the evening?"
  "- How many people live in the household?"
  "- Are there air conditioning units? If so, how many?"
  "- Is there a swimming pool or an electric vehicle?"
    " If the user is not located in Martinique. Tell him that for the moment we are only present in Martinique. We will be able to provide you with the best service in the future."
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
    Notes: <Brief summary in french of the user's questions and profile based on the entire conversation>
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
            if extracted_phone and len(
                    extracted_phone) >= 10:  
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

def cleanup_sessions(timeout=3600*6):  # Timeout in seconds
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
        await asyncio.sleep(3600*6)  
        clear_cache()
        cleanup_sessions()


@app.on_event("startup")
async def startup_event():
    # Start the background task when the FastAPI app starts
    asyncio.create_task(periodic_reset())
    #logger.info("Started background tasks")


@app.post("/chat")
async def chat(request: Request):
    input_data = await request.json()
    input_text = input_data.get("input")
    session_id = input_data.get("session_id")  
    if not input_text:
        #logger.warning("No input provided in the request.")
        return {"error": "No input provided"}
    if not session_id:
        #logger.warning("No session_id provided in the request.")
        return {"error": "No session_id provided"}  # Ensure session_id is present

    # Update last activity time
    last_active_time[session_id] = time.time()

    logger.info(f"Query received: {input_text}")

    if session_id not in conversation_memory_dict:
        conversation_memory_dict[session_id] = ConversationBufferMemory()
        #logger.info(f"New conversation memory initialized for session {session_id}")

    conversation_memory = conversation_memory_dict[session_id]

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

    try:
        if exchange_count[user_id] < 3:
            persona_prompt = initial_persona_prompt

        elif info_collected[user_id]:
            persona_prompt = faq
            logger.info("FAQ")
        else:
            persona_prompt = salesman_persona_prompt

        #logger.info(f"Conversation memory for session {session_id} before update: {conversation_memory.buffer}")
        
        context = retriever.get_relevant_documents(input_text)
        context_text = "\n\n".join([doc.page_content for doc in context])
        question_with_context = f"{persona_prompt}\n\nContext:\n{context_text}\n\nQuestion: {input_text}"

        conversation_response = conversation_chain.run(question_with_context)
        #logger.info(f"LLM response for conversation: {conversation_response}")
        # Manually save the context in the memory
        conversation_memory.save_context({"input": input_text}, {"output": conversation_response})

        #logger.info(f"Conversation memory for session {session_id} after update: {conversation_memory.buffer}")

        #logger.info(f"Exchange count: {exchange_count[user_id]}")

        # If the salesman persona is active, check if it was already given
        if persona_prompt == salesman_persona_prompt:
            if not salesman_prompt_given[user_id]:
                # First time showing the salesman prompt, just return it without analyzing the input
                salesman_prompt_given[user_id] = True
                return {"answer": conversation_response}
            else:
                # Salesman prompt was already given, analyze the user's response for missing information
                conversation_history = conversation_memory.buffer
                #logger.info(f"Accessing conversation memory for session {session_id}: {conversation_history}")
                # Analyze user input and conversation history to extract missing information
                fullname, email, phone, notes, detect = analyze_input_for_information(
                    input_text, conversation_history, user_id)

                # Update the user data with any new information
                user_data[user_id] = {
                    "fullname": fullname,
                    "email": email,
                    "phone": phone,
                    "notes": notes
                }

                if detect:
                    #logger.info(   f"All required information collected for user {user_id}: {user_data[user_id]}")

                    if send_to_airtable(user_data[user_id]["fullname"],
                                        user_data[user_id]["email"],
                                        user_data[user_id]["phone"],
                                        user_data[user_id]["notes"]):

                        info_collected[user_id] = True
                        return {
                            "answer": {conversation_response}
                        }
                    else:
                        logger.error(
                            "Failed to send information to Google Sheets.")
                        return {"answer": conversation_response}

                return {"answer": conversation_response}

        return {"answer": conversation_response}

    except Exception as e:
        logger.error(f"Error during chat invocation: {e}")
        return {"error": str(e)}


@app.get("/test-airtable")
def test_airtable():
    test_fullname = "Test User"
    test_email = "testuser@example.com"
    test_phone = "1234567890"
    test_notes = "This is a test note."

    #logger.info(f"Attempting to send data to Airtable: Full Name: {test_fullname}, Email: {test_email}, Phone: {test_phone}, Notes: {test_notes}")

    #if not AIRTABLE_PERSONAL_TOKEN:
        #logger.error("Airtable Personal Access Token is missing!")
    #else:
        #logger.info(f"Airtable Personal Access Token (first 10 chars): {AIRTABLE_PERSONAL_TOKEN[:10]}...")

    #if not AIRTABLE_BASE_ID:
        #logger.error("Airtable Base ID is missing!")
    #else:
        #logger.info(f"Airtable Base ID: {AIRTABLE_BASE_ID}")

    #if not AIRTABLE_TABLE_NAME:
        #logger.error("Airtable Table Name is missing!")
    #else:
        #logger.info(f"Airtable Table Name: {AIRTABLE_TABLE_NAME}")
    success = send_to_airtable(test_fullname, test_email, test_phone, test_notes)

    if success:
        logger.info("Data successfully sent to Airtable.")
        return {"message": "Test data sent to Airtable successfully!"}
    else:
        logger.error("Failed to send data to Airtable.")
        return {"error": "Failed to send test data to Airtable."}


@app.get("/test-google-sheets")
def test_google_sheets():
    test_fullname = "Test User"
    test_email = "testuser@example.com"
    test_phone = "1234567890"
    test_notes = "This is a test note."

    success = send_to_google_sheet(test_fullname, test_email, test_phone,
                                   test_notes)

    if success:
        return {"message": "Test data sent to Google Sheets successfully!"}
    else:
        return {"error": "Failed to send test data to Google Sheets."}


#curl http://localhost:8000/test-google-sheets

if __name__ == "__main__":
    logger.info("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))
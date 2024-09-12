import os
import time
import json
import hashlib
from fastapi import FastAPI, Request
import uvicorn
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

# Set up the language model (using GPT-3.5-turbo)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define retriever as a global variable
retriever = None

# Google Sheets credentials - replace with your actual values
GOOGLE_SHEET_ID = "10oAG2URrrX1YJr0StNAXszFycYPwBMmj9N6Y18TwcVk"  # Your Google Sheet ID
GOOGLE_SHEET_RANGE = "Sheet1!A1:C1"  # Example range, adjust according to your needs

# Service Account JSON contents
# Load the service account from Replit secret
service_account_info = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

# Parse the JSON from the environment variable
if service_account_info:
    service_account_info = json.loads(service_account_info)

    # Create the service account credentials
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
else:
    raise EnvironmentError("Service account info is missing in the environment variables")
    
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

logger.info(f"GOOGLE_SHEET_ID: {GOOGLE_SHEET_ID}, GOOGLE_SHEET_RANGE: {GOOGLE_SHEET_RANGE}")

# Function to analyze negativity and update the 'detect' variable
def analyze_negativity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity > 0  # Positive polarity indicates valid information

# Function to send data to Google Sheets
def send_to_google_sheet(fullname, email, phone, notes):
    logger.info("Triggered send_to_google_sheet function.")
    logger.info(f"Data to send: Full Name: {fullname}, Email: {email}, Phone: {phone}, Notes: {notes}")

    values = [[fullname, email, phone, notes]]
    body = {'values': values}

    try:
        result = sheet.values().append(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=GOOGLE_SHEET_RANGE,  # Adjust the range to include the phone column
            valueInputOption="RAW",
            body=body
        ).execute()

        if result:
            logger.info("Data successfully sent to Google Sheets.")
            return True
        else:
            logger.error("Failed to send data to Google Sheets.")
            return False
    except Exception as e:
        logger.error(f"Exception occurred while sending data to Google Sheets: {e}")
        return False

# File paths and constants
DOCUMENT_PATH = "docs/Solar_guide.pdf"
HASH_FILE = "document_hash.txt"
VECTOR_STORE_PATH = "faiss_vector_db"

# Initialize Conversation Buffer Memory
conversation_memory = ConversationBufferMemory()

# Function to calculate the hash of the document
def calculate_file_hash(filepath):
    hash_algo = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while chunk := file.read(8192):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()

# Function to check if the document has changed
def document_changed():
    new_hash = calculate_file_hash(DOCUMENT_PATH)

    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, 'r') as f:
            old_hash = f.read()

        if new_hash == old_hash:
            return False  # Document hasn't changed

    # Document has changed or HASH_FILE doesn't exist
    with open(HASH_FILE, 'w') as f:
        f.write(new_hash)

    return True

# Function to load and ingest documents
def ingest_docs():
    # Load the document
    loader = UnstructuredPDFLoader(DOCUMENT_PATH)
    docs = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings()

    # Check if chunks are created correctly
    if not chunks:
        raise ValueError("No chunks created from the document. Check the document loading process.")

    # Create or overwrite the vector store
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)

    print("Document successfully ingested into knowledge base")

    return vectorstore  # Return the vectorstore to update retriever globally


# Check if document has changed and ingest if needed
if document_changed():
    print("Document changed. Ingesting new document...")
    conversation_memory.clear()
    conversation_memory = ConversationBufferMemory()
    print("Conversation memory cleared")
    # Ingest the new documents and update the retriever
    vectorstore = ingest_docs()
    retriever = vectorstore.as_retriever()  # Update the global retriever after ingestion
else:
    print("Document unchanged. Loading existing vector store.")
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()  # Set retriever during initialization

# Initialize the Conversation Chain with memory
conversation_chain = ConversationChain(
    llm=llm,
    memory=conversation_memory,
)


# Initialize FastAPI application
app = FastAPI(
    title="LangChain Server with Knowledge Base",
    version="1.0",
    description=
    "A knowledge-based API server using LangChain and Solar Guide as a knowledge source",
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the root endpoint!"}

# Define the two persona prompts
initial_persona_prompt = (    
"You are a friendly representative of Kay Soley, knowledgeable about solar energy. Answer in the same language as the user. Don't say Hello. Your goal is to engage in a natural conversation, and answer based on the Solar Guide any questions the user may have. Do not ask for personal information at this stage.\n If a question cannot be asnwered by the content of the Solar Guide, say that you are unsure and that the user should ask this question to one of our Technicians during a telephone or home appointment.\n Clarity and Conciseness: Use bullet points or numbered lists for clarity in your responses, and keep responses concise, limited to 2-3 sentences."
)

salesman_persona_prompt = (
    "You are a friendly and persuasive solar energy salesman working for Kay Soley. Answer in the same language as the user. "
    "Don't say Hello. Your goal is to engage in a natural conversation with the user, subtly gather their full name, email address, phone number, and any specific needs or questions they may have about solar energy and Kay Soley with brief answers. "
    "Clarity and Conciseness: Use bullet points or numbered lists for clarity in your responses, and keep responses concise, limited to 2-3 sentences.\n"
    "- Use a polite and non-intrusive approach when asking for the user's full name, phone number and email address.\n"
    "- Ensure the conversation feels natural and the user feels comfortable providing their information.\n\n"
    "Desired Format:\n"
    "- Gather the following information from the user:\n"
    "  - Full Name: <extracted full name if any>\n"
    "  - Email: <extracted email address if any>\n"
    "  - Phone: <extracted phone number if any>\n"
    "  - Notes: <Brief and precise summary of the user's needs or questions>"
)

# Dictionaries to track user state
exchange_count = {}
user_data = {}
info_collected = {}
salesman_prompt_given = {}

def analyze_input_for_information(input_text, conversation_history, user_id):
    """Analyzes the user's input and full conversation history to check for personal information and create a summary."""

    verification_prompt = f"""
    You are to analyze the following conversation and input to verify if it contains the user's full name, email address, phone number, and provide a brief summary of their needs, intent, or questions based on the entire conversation.

    Conversation:
    {conversation_history}

    Latest Input:
    {input_text}

    Please respond with the information found in the following format:
    Full Name: <extracted full name if any>
    Email: <extracted email address if any>
    Phone: <extracted phone number if any>
    Notes: <Brief and precise summary of the user's intent and profile based on the entire conversation>
    """

    # Get the response from the LLM for verification and summarization
    verification_result = llm.invoke(verification_prompt)

    extracted_data = verification_result.content.strip().split("\n")

    fullname = user_data.get(user_id, {}).get('fullname', 'N/A')
    email = user_data.get(user_id, {}).get('email', 'N/A')
    phone = user_data.get(user_id, {}).get('phone', 'N/A')
    notes = "N/A"
    detect = False

    # Parsing the extracted data for name, email, phone, and notes
    for item in extracted_data:
        if item.startswith("Full Name:"):
            extracted_fullname = item.replace("Full Name:", "").strip()
            if extracted_fullname and extracted_fullname.lower() != "not provided":
                fullname = extracted_fullname
        elif item.startswith("Email:"):
            extracted_email = item.replace("Email:", "").strip()
            if extracted_email and "@" in extracted_email:  # Simple email pattern detection
                email = extracted_email
        elif item.startswith("Phone:"):
            extracted_phone = item.replace("Phone:", "").strip()
            if extracted_phone and len(extracted_phone) >= 10:  # Basic phone number validation
                phone = extracted_phone
        elif item.startswith("Notes:"):
            extracted_notes = item.replace("Notes:", "").strip()
            if extracted_notes and extracted_notes.lower() != "not provided":
                notes = extracted_notes

    # Store the collected information in the user_data dictionary
    user_data[user_id] = {"fullname": fullname, "email": email, "phone": phone, "notes": notes}

    # Check if all required information is collected
    detect = all(value != "N/A" for value in user_data[user_id].values())

    return fullname, email, phone, notes, detect

async def reset_exchange_count():
    global exchange_count  # Declare exchange_count as global to modify the global variable
    while True:
        await asyncio.sleep(120)  # Wait for 10 minutes (600 seconds)
        exchange_count.clear()  # Clear the exchange count for all users
        print("Exchange counts reset.")

# Function to clear the .cache directory
def clear_cache():
    cache_path = ".cache/*"  # Path to the cache directory
    try:
        os.system(f"rm -rf {cache_path}")
        logger.info(".cache directory has been cleared.")
    except Exception as e:
        logger.error(f"Error while clearing cache: {e}")

# Function to clear both conversation memory and cache
def reset_conversation_and_cache():
    global conversation_memory
    conversation_memory.clear()
    logger.info("Conversation memory has been cleared.")

    # Clear cache directory
    clear_cache()

# Background task to clear memory and cache every 20 minutes
async def periodic_reset():
    while True:
        await asyncio.sleep(20 * 60)  # Wait for 20 minutes
        reset_conversation_and_cache()
        

@app.on_event("startup")
async def startup_event():
    # Start the background task when the FastAPI app starts
    asyncio.create_task(periodic_reset())
    asyncio.create_task(reset_exchange_count())
    logger.info("Started background tasks")

@app.post("/chat")
async def chat(request: Request):
    input_data = await request.json()
    input_text = input_data.get("input")

    if not input_text:
        logger.warning("No input provided in the request.")
        return {"error": "No input provided"}

    logger.info(f"Query received: {input_text}")

    # Use a dynamic user ID if needed, or replace with proper authentication logic
    user_id = "user_demo"

    # Initialize tracking for user if not already done
    if user_id not in exchange_count:
        exchange_count[user_id] = 0
    if user_id not in info_collected:
        info_collected[user_id] = False
    if user_id not in user_data:
        user_data[user_id] = {
            "fullname": "N/A",
            "email": "N/A",
            "phone": "N/A",
            "notes": "N/A"
        }
    if user_id not in salesman_prompt_given:
        salesman_prompt_given[user_id] = False  # Track if salesman prompt has been given

    # Increment exchange count for the user
    exchange_count[user_id] += 1

    try:
        # Determine whether to use the initial or salesman persona prompt based on the conversation stage
        if info_collected[user_id] or exchange_count[user_id] < 3:
            persona_prompt = initial_persona_prompt
        else:
            persona_prompt = salesman_persona_prompt

        # Retrieve relevant knowledge from the vectorstore
        context = retriever.get_relevant_documents(input_text)
        context_text = "\n\n".join([doc.page_content for doc in context])
        question_with_context = f"{persona_prompt}\n\nContext:\n{context_text}\n\nQuestion: {input_text}"

        # Get the response from the LLM using the conversation chain
        conversation_response = conversation_chain.run(question_with_context)
        logger.info(f"LLM response for conversation: {conversation_response}")

        # If the salesman persona is active, check if it was already given
        if persona_prompt == salesman_persona_prompt:
            if not salesman_prompt_given[user_id]:
                # First time showing the salesman prompt, just return it without analyzing the input
                salesman_prompt_given[user_id] = True
                return {"answer": conversation_response}
            else:
                # Salesman prompt was already given, analyze the user's response for missing information
                conversation_history = conversation_memory.buffer

                # Analyze user input and conversation history to extract missing information
                fullname, email, phone, notes, detect = analyze_input_for_information(input_text, conversation_history, user_id)

                # Update the user data with any new information
                user_data[user_id] = {
                    "fullname": fullname,
                    "email": email,
                    "phone": phone,
                    "notes": notes
                }

                if detect:
                    logger.info(f"All required information collected for user {user_id}: {user_data[user_id]}")

                    # Send the collected information to Google Sheets
                    if send_to_google_sheet(user_data[user_id]["fullname"],
                                            user_data[user_id]["email"],
                                            user_data[user_id]["phone"],
                                            user_data[user_id]["notes"]):
                        # Mark info as collected and reset for this user
                        info_collected[user_id] = True
                        return {
                            "answer": 
                            "Merci beaucoup ! Nous vous contacterons sous peu."
                        }
                    else:
                        logger.error("Failed to send information to Google Sheets.")
                        return {"answer": conversation_response}

                # If all data is not yet collected, continue asking for missing data
                #missing_info_response = ""
                #if user_data[user_id]["fullname"] == "N/A":
                #    missing_info_response += "Could you please provide your full name? "
                #if user_data[user_id]["email"] == "N/A":
                #    missing_info_response += "Can you share your email address? "
                #if user_data[user_id]["phone"] == "N/A":
                #    missing_info_response += "I still need your phone number, if you don't mind. "

                return {"answer": conversation_response}

        # Return the response if not in salesman mode
        return {"answer": conversation_response}

    except Exception as e:
        logger.error(f"Error during chat invocation: {e}")
        return {"error": str(e)}

@app.get("/test-google-sheets")
def test_google_sheets():
    test_fullname = "Test User"
    test_email = "testuser@example.com"
    test_phone = "1234567890"
    test_notes = "This is a test note."

    success = send_to_google_sheet(test_fullname, test_email, test_phone, test_notes)

    if success:
        return {"message": "Test data sent to Google Sheets successfully!"}
    else:
        return {"error": "Failed to send test data to Google Sheets."}
        
#curl http://localhost:8000/test-google-sheets


if __name__ == "__main__":
    logger.info("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

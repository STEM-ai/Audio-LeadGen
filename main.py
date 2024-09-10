import os
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

# Set up the language model (using GPT-3.5-turbo)
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Google Sheets credentials - replace with your actual values
GOOGLE_SHEET_ID = "10oAG2URrrX1YJr0StNAXszFycYPwBMmj9N6Y18TwcVk"  # Your Google Sheet ID
GOOGLE_SHEET_RANGE = "Sheet1!A1:C1"  # Example range, adjust according to your needs

# Service Account JSON contents
SERVICE_ACCOUNT_INFO = {
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
}

# Set up Google Sheets API client
creds = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
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
    logger.info("Starting document ingestion process.")
    loader = UnstructuredPDFLoader(DOCUMENT_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
    vectorstore.save_local(VECTOR_STORE_PATH)
    logger.info("Document successfully ingested into knowledge base")
    print("Document successfully ingested into knowledge base")

# Check if document has changed and ingest if needed
if document_changed():
    logger.info("Document changed. Ingesting new document...")
    conversation_memory.clear()
    ingest_docs()
else:
    logger.info("Document unchanged. Loading existing vector store.")

vectorstore = FAISS.load_local(VECTOR_STORE_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

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


# Function to reset conversation memory
def reset_conversation_memory():
    global conversation_memory
    conversation_memory.clear()
    logger.info("Conversation memory has been cleared.")

# Background task to reset conversation memory every 20 minutes
async def periodic_memory_reset():
    while True:
        await asyncio.sleep(20 * 60)  # Wait for 20 minutes
        reset_conversation_memory()

@app.on_event("startup")
async def startup_event():
    # Start the background task when the FastAPI app starts
    asyncio.create_task(periodic_memory_reset())
    logger.info("Started background task to reset conversation memory every 20 minutes.")



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



if __name__ == "__main__":
    logger.info("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

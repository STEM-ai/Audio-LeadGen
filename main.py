import os
from fastapi import FastAPI, Request
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import requests
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
def send_to_google_sheet(fullname, email, notes):
    logger.info("Triggered send_to_google_sheet function.")
    logger.info(f"Data to send: Full Name: {fullname}, Email: {email}, Notes: {notes}")

    values = [[fullname, email, notes]]
    body = {'values': values}

    try:
        result = sheet.values().append(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=GOOGLE_SHEET_RANGE,
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

# Function to load and ingest documents
def ingest_docs():
    logger.info("Starting document ingestion process.")
    loader = UnstructuredPDFLoader("docs/Solar_guide.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=chunks,
                                       embedding=OpenAIEmbeddings())
    vectorstore.save_local("faiss_vector_db")
    logger.info("Document successfully ingested into knowledge base")
    print("Document successfully ingested into knowledge base")

if not os.path.isdir("faiss_vector_db"):
    logger.info("Vector database not found. Ingesting documents...")
    print("Vector database not found. Ingesting documents...")
    ingest_docs()

vectorstore = FAISS.load_local("faiss_vector_db",
                               OpenAIEmbeddings(),
                               allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Initialize Conversation Buffer Memory
conversation_memory = ConversationBufferMemory()

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
"You are a friendly representative of Kay Soley, knowledgeable about solar energy. Your goal is to engage in a natural conversation, and answer based on the Solar Guide any questions the user may have. If a question cannot be asnwered by the content of the Solar Guide, answer following the industry's standards. Do not ask for personal information at this stage."
)

salesman_persona_prompt = (
    "You are a friendly and persuasive solar energy salesman working for Kay Soley. Your goal is to engage in a natural conversation with the user,       subtly gather their full name, email address, and any specific needs or questions they may have about solar energy and Kay Soley with brief answers\n"
    "Instructions:\n"
    "- Begin by building rapport and expressing enthusiasm for helping the user learn about solar energy.\n"
    "- Ask open-ended questions to understand the user's interest in solar energy and gently guide the conversation towards gathering their contact information.\n"
    "- Offer relevant information about Kay Soley's solar energy solutions and how they can meet the user's needs.\n"
    "- Use a polite and non-intrusive approach when asking for the user's full name and email address.\n"
    "- Ensure the conversation feels natural and the user feels comfortable providing their information.\n\n"
    "Desired Format:\n"
    "- Gather the following information from the user:\n"
    "  - Full Name: <extracted full name if any>\n"
    "  - Email: <extracted email address if any>\n"
    "  - Notes: <Brief summary of the conversation and user's needs or questions>"
)

# Dictionaries to track user state
exchange_count = {}
info_collected = {}

def analyze_input_for_information(input_text):
    """Analyzes the user's input to check for personal information."""
    verification_prompt = f"""
    You are to analyze the following conversation to verify if it contains the user's full name, email address, and a brief summary of their needs or questions.

    Conversation:
    {input_text}

    Please respond with the information found in the following format:
    Full Name: <extracted full name if any>
    Email: <extracted email address if any>
    Notes: <Brief summary of the conversation>
    """

    # Get the response from the LLM specifically for verification
    verification_result = llm.invoke(verification_prompt)

    # Print the verification result for debugging purposes
    print(f"Verification LLM Output: {verification_result.content}")

    extracted_data = verification_result.content.strip().split("\n")

    fullname = "N/A"
    email = "N/A"
    notes = "N/A"
    detect = False

    # Improve detection logic by looking for common name and email patterns
    for item in extracted_data:
        if item.startswith("Full Name:"):
            extracted_fullname = item.replace("Full Name:", "").strip()
            if extracted_fullname and extracted_fullname.lower() != "not provided":
                fullname = extracted_fullname
                detect = True
        elif item.startswith("Email:"):
            extracted_email = item.replace("Email:", "").strip()
            if extracted_email and "@" in extracted_email:  # Simple email pattern detection
                email = extracted_email
                detect = True
        elif item.startswith("Notes:"):
            extracted_notes = item.replace("Notes:", "").strip()
            if extracted_notes and extracted_notes.lower() != "not provided":
                notes = extracted_notes
                detect = True

    return fullname, email, notes, detect

@app.post("/chat")
async def chat(request: Request):
    input_data = await request.json()
    input_text = input_data.get("input")

    if not input_text:
        logger.warning("No input provided in the request.")
        return {"error": "No input provided"}

    logger.info(f"Query received: {input_text}")

    # Get user ID to track exchange count (for demo purposes, using a placeholder user_id)
    user_id = "user_demo"  # Replace this with actual user ID tracking logic if needed
    exchange_count[user_id] = exchange_count.get(user_id, 0) + 1
    info_collected[user_id] = info_collected.get(user_id, False)

    try:
        # Determine which persona prompt to use based on the exchange count and whether info has been collected
        if info_collected[user_id]:
            persona_prompt = initial_persona_prompt
        elif exchange_count[user_id] < 3:
            persona_prompt = initial_persona_prompt
        else:
            persona_prompt = salesman_persona_prompt

        # Retrieve relevant knowledge from the vectorstore
        context = retriever.get_relevant_documents(input_text)
        context_text = "\n\n".join([doc.page_content for doc in context])
        # Combine the persona prompt, context, and user query
        question_with_context = f"{persona_prompt}\n\nContext:\n{context_text}\n\nQuestion: {input_text}"

        # Get the response from the LLM using the conversation chain
        conversation_response = conversation_chain.run(question_with_context)
        logger.info(f"LLM response for conversation: {conversation_response}")

        # If this is the third exchange or later, analyze user input for personal information and negativity
        if exchange_count[user_id] >= 3 and not info_collected[user_id]:
            fullname, email, notes, detect = analyze_input_for_information(input_text)

            if detect:
                collected_info = {"fullname": fullname, "email": email, "notes": notes}
                logger.info(f"Extracted Information - Full Name: {fullname}, Email: {email}, Notes: {notes}")

                # If all information is collected and valid, send it to Google Sheets
                if all(value != "N/A" for value in collected_info.values()):
                    logger.info("All required information collected and valid. Sending to Google Sheets...")
                    if send_to_google_sheet(collected_info["fullname"],
                                            collected_info["email"],
                                            collected_info["notes"]):
                        # Mark information as collected and reset to initial persona prompt
                        info_collected[user_id] = True
                        return {
                            "answer": 
                            "Thank you! Your information has been collected successfully. We'll be in touch soon."
                        }
                    else:
                        logger.error("Failed to send information to Google Sheets.")
                        return {"answer": conversation_response}

        # If not all information is collected, send the first GPT response directly to Voiceflow
        return {"answer": conversation_response}


    except Exception as e:
        logger.error(f"Error during chat invocation: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

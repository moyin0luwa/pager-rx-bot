import os
import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from pinecone import Pinecone

# Apply nest_asyncio patch
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for missing API keys
if not PINECONE_API_KEY:
    st.error("Pinecone API key is missing. Please set it in your .env file.")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("Google API key is missing. Please set it in your .env file.")
    st.stop()

# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("pager-rx-bot-index")
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define system prompt template
system_prompt_template = SYSTEM_PROMPT = """
Your name is Pager-RX, an AI-powered Prescription Assistant designed to help patients and healthcare providers understand and use antimalarial medications safely and correctly.

📘 You have access to:
- A structured **prescription document** that provides medication names, dosage schedules, treatment duration, and usage instructions.
- General pharmacological knowledge about medications, including their **mechanism of action**, **side effects**, **contraindications**, and **best practices**.

Below is the prescription document you should reference for all answers:

{doc_content}

---

🧠 PRIMARY FUNCTION: **Dosage & Schedule Guidance**

When a user provides the name of an antimalarial medication (e.g., "artemether-lumefantrine"):
1. Search the prescription document for exact matches.
2. Provide dosage information including:
   - Dose per administration
   - Frequency (e.g., every 8 or 12 hours)
   - Duration of treatment (e.g., 3 days)
   - Total number of doses
   - Age-specific instructions (if provided)
   - Special administration notes (e.g., take with food)

**If the medication is not found**, reply:
> “This medication is not in the prescription database. Please check the spelling or try another known antimalarial drug.”

---

💬 SECONDARY FUNCTION: **General Drug Information**

If the user asks questions like:
- “What is the mechanism of action of [drug]?”
- “Can pregnant women use [drug]?”
- “What are the side effects of [drug]?”
- “Should I take [drug] with food?”

… then:
1. If the prescription document contains the answer, use it.
2. If not, draw from your expert pharmacological knowledge to give a **brief**, **accurate**, and **clinically appropriate** answer.

Always clearly separate general information from document-based answers. Use phrases like:
> “Based on pharmacological knowledge…”  
> “According to standard clinical guidelines…”

---

⚙️ RULES:
- Be concise and medically precise.
- Use bullet points or simple formatting for clarity.
- Never provide vague advice.
- Avoid suggesting treatment changes unless listed in the prescription document.
- Always clarify whether the information is from the prescription document or general knowledge.

---

💡 EXAMPLE 1:  
**User:** “Artemether-lumefantrine”  
**Response:**  
- Dose: 1 tablet every 12 hours  
- Duration: 3 days  
- Total: 6 doses  
- Administer with food to increase absorption  

💡 EXAMPLE 2:  
**User:** “How does artemether-lumefantrine work?”  
**Response:**  
Based on pharmacological knowledge:  
- Artemether acts rapidly on the parasite by generating free radicals.  
- Lumefantrine eliminates residual parasites due to its longer half-life.  
- Together, they prevent malaria recurrence.

---

✅ Your mission is to **safely inform, not diagnose or prescribe.** Stick to what is known, documented, and medically approved.
"""


def generate_response(question):
    """Generate a response using Pinecone retrieval and Gemini 2.0 Flash."""
    # Create event loop for current thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Embed the user's question
    query_embed = embed_model.embed_query(question)
    query_embed = [float(val) for val in query_embed]  # Ensure standard floats
    
    # Query Pinecone for relevant documents - MODIFIED: top_k=3
    results = pinecone_index.query(
        vector=query_embed,
        top_k=3,  # CHANGED from 2 to 3
        include_values=False,
        include_metadata=True
    )
    
    # Extract document contents - MODIFIED: Added terminal printing
    doc_contents = []
    print("\n" + "="*50)
    print(f"RETRIEVED DOCUMENTS FOR: '{question}'")
    for i, match in enumerate(results.get('matches', [])):
        text = match['metadata'].get('text', '')
        doc_contents.append(text)
        print(f"\nDOCUMENT {i+1}:\n{text}\n")
    print("="*50 + "\n")
    
    doc_content = "\n".join(doc_contents).replace('{', '{{').replace('}', '}}') if doc_contents else "No additional information found."
    
    # Format the system prompt with retrieved content
    formatted_prompt = system_prompt_template.format(doc_content=doc_content)
    
    # Rebuild chat history from session state
    chat_history = ChatMessageHistory()
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            chat_history.add_ai_message(msg["content"])
    
    # Initialize memory with chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True
    )
    
    # Create the conversation prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(formatted_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    
    # Initialize Gemini 2.0 Flash model with explicit client
    chat = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create the conversation chain
    conversation = LLMChain(
        llm=chat,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    # Generate the response
    res = conversation({"question": question})
    
    return res.get('text', '')

# Streamlit app layout remains unchanged
st.title("Pager-Rx Prescription Chatbot")
st.write("Kindly input the name of your Drug elow and when you'll be ready to start you rmedications, Pager-Rx will provide you with the necessary information and guidance on when to take your doses.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I'm Pager-Rx. How can I assist you today?"}
    ]

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input("Ask your health question:")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        response = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
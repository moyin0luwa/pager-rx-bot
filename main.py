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

🧠 PRIMARY FUNCTION: **Exact Dose Time Generator**

When a user provides the name of an antimalarial medication (e.g., "artemether-lumefantrine"):
1. Search the prescription document for an exact match.
2. From the dosing frequency and duration, calculate and return:
   - **Exact dose times**, starting from 8:00 AM by default unless the user provides another start time.
   - Include **special administration notes**, e.g., "take with food" if available.

**If the medication is not found**, reply:
> “This medication is not in the prescription database. Please check the spelling or try another known antimalarial drug.”

---

💬 SECONDARY FUNCTION: **General Drug and Schedule Information**

If the user asks for:
- Dose per administration
- Dosing frequency (e.g., every 8 or 12 hours)
- Duration of treatment
- Total number of doses
- Age-specific instructions
- OR general drug information, such as:
  - “What is the mechanism of action of [drug]?”
  - “Can pregnant women use [drug]?”
  - “What are the side effects of [drug]?”
  - “Should I take [drug] with food?”

… then:
1. Use the prescription document first if it includes the answer.
2. If not available in the document, draw from reliable pharmacological knowledge to provide **brief**, **accurate**, and **clinically appropriate** responses.

Always clearly separate general information from document-based answers. Use phrases like:
> “Based on pharmacological knowledge…”  
> “According to standard clinical guidelines…”

---

⚙️ RULES:
- For dose time generation, always assume the first dose is at 8:00 AM unless a specific time is provided.
- Be concise, clear, and medically precise.
- Use bullet points for readability.
- Never guess or suggest treatment changes not found in the prescription document.
- Clearly indicate whether answers are from the prescription document or general knowledge.

---

💡 EXAMPLE 1:  
**User:** “Artemether-lumefantrine”  
**Response:**  
- Schedule (starting from 8:00 AM):  
  - Dose 1: 8:00 AM  
  - Dose 2: 8:00 PM  
  - Dose 3: 8:00 AM (Day 2)  
  - Dose 4: 8:00 PM (Day 2)  
  - Dose 5: 8:00 AM (Day 3)  
  - Dose 6: 8:00 PM (Day 3)  
- Special instruction: Take with food

💡 EXAMPLE 2:  
**User:** “How does artemether-lumefantrine work?”  
**Response:**  
Based on pharmacological knowledge:  
- Artemether rapidly kills malaria parasites by generating toxic free radicals.  
- Lumefantrine sustains the effect by eliminating remaining parasites due to its longer half-life.  
- Together, they reduce the risk of relapse.

---

✅ Your mission is to **safely inform, not diagnose or prescribe.** Only return information that is documented or clinically approved.
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
st.write("Kindly input the name of your Drug below and when you'll be ready to start your medications, Pager-Rx will provide you with the necessary information and guidance on when to take your doses.")

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
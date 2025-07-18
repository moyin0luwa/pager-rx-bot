# Pager-RX 🤖💊  
**Intelligent Prescription Guidance Chatbot**

Welcome to the **Pager-RX** project! 🎉  
This prototype is a prescription management chatbot that uses **Large Language Models (LLMs)**, **vector databases**, and **Streamlit** to deliver intelligent, real-time medication guidance.

**GitHub Repository**: [github.com/moyin0luwa/pager-rx-bot](https://github.com/moyin0luwa/pager-rx-bot)

---

## 🔍 About The Project

**Pager-RX** is a chatbot designed to help **patients** and **healthcare professionals** access accurate prescription usage instructions. It fetches context-rich data from a structured knowledge base (e.g., formularies, standard prescription sheets), delivering user-friendly answers powered by cutting-edge AI.

### 🎯 Core Capabilities

- Accepts natural language questions related to medication or prescriptions.
- Retrieves relevant prescription data from a **Pinecone vector database**.
- Uses **Google Gemini 2.0 Flash** to generate safe and accurate responses.
- Offers **multi-turn conversation memory** with Langchain.
- Runs on an intuitive **Streamlit interface** for quick testing and demos.

---

## ✨ Features

| Feature                        | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| 💡 AI-Powered Guidance         | Responses are generated using **Google Gemini 2.0 Flash** for accuracy.     |
| 📚 PDF-Driven Knowledge Base  | Prescription content is embedded using **GoogleGenerativeAIEmbeddings**.    |
| 🧠 Contextual Memory           | Multi-turn conversations supported via **Langchain memory modules**.       |
| ⚡ Interactive UI              | Simple and responsive UI built using **Streamlit**.                         |
| 🧪 Modular Architecture        | Codebase inspired by the **DSA_HIV** project for readability and learning. |

---

## 🛠️ Tech Stack

| Component               | Technology Used                              |
|------------------------|-----------------------------------------------|
| Language               | Python                                        |
| UI Framework           | [Streamlit](https://streamlit.io)             |
| LLM Framework          | [Langchain](https://www.langchain.com)        |
| Embedding Provider     | GoogleGenerativeAIEmbeddings                  |
| Vector Store           | [Pinecone](https://www.pinecone.io)           |
| LLM                    | Google Gemini 2.0 Flash                       |
| Secrets Management     | `python-dotenv` for `.env` loading            |

Pager-RX: Intelligent Prescription Guidance Chatbot

Welcome to the Pager-RX project! 🎉 This prototype demonstrates how to build an intelligent prescription management chatbot using Large Language Models (LLMs), vector databases, and Streamlit for an intuitive user interface.

This README will guide you step-by-step from setup to understanding the core components of this application. Whether you're new to Python, LLMs, or full-stack development, this project provides a practical learning path.

GitHub Repository: https://github.com/moyin0luwa/pager-rx-bot

🔍 About The Project

Pager-RX is a Prescription Management Chatbot designed to guide patients and healthcare practitioners through medication usage instructions. It pulls relevant prescription data from a structured knowledge base to deliver clear, concise, and accurate instructions.

The chatbot:

Accepts user questions related to prescriptions.

Searches a knowledge base (PDF/text embeddings) stored in Pinecone for relevant data.

Uses Google Gemini 2.0 Flash to generate a context-aware response.

Returns a brief but accurate dosage or usage recommendation.

✨ Features

AI-Powered Guidance: Built with Google Gemini 2.0 Flash for intelligent responses.

PDF-Driven Knowledge Base: Converts structured prescription data (e.g., from formularies) into searchable embeddings.

Contextual Memory: Supports multi-turn conversations using Langchain memory modules.

Interactive UI: Built with Streamlit for rapid deployment and ease-of-use.

Educational Structure: Easily understandable code and structure, based on the DSA_HIV project.

🛠️ Tech Stack

Python: Primary language.

Streamlit: Frontend interface.

Langchain: LLM framework for chaining and memory.

Google Gemini 2.0 Flash: LLM for generation.

Pinecone: Vector database for semantic search.

GoogleGenerativeAIEmbeddings: For embedding prescription data.

Dotenv: For secure API key storage.

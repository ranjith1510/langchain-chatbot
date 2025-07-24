# langchain-chatbot

LangChain FAISS Chatbot

Description:
This project implements a conversational chatbot using LangChain, FAISS vector store, and OpenAI embeddings. It loads and splits documents from a web page, creates local embeddings, stores them in FAISS, and uses OpenAI's Chat API to answer queries interactively.

Code Structure:
- main.py        : Main Python script containing the chatbot implementation.
- requirements.txt : Lists required Python packages.

Setup:
1. Install Python 3.8 or higher.
2. Install dependencies:
   pip install -r requirements.txt
3. Set your OpenAI API key inside main.py or as environment variable OPENAI_API_KEY.

Usage:
Run the chatbot with:
   python main.py
Type your question and get answers from the loaded documents.
Type 'exit' or 'quit' to stop.

requirements.txt content:
langchain
langchain-community
sentence-transformers
faiss-cpu
langchain-openai

Sample Input/Output:

You: What is LangChain used for?
🤖 Bot: LangChain is a framework designed to simplify the development of applications powered by language models...

You: exit
Goodbye!


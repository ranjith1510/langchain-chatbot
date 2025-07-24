# Install required packages (run once)
!pip install -U langchain langchain-community sentence-transformers faiss-cpu langchain-openai

import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = ""

print("Loading documents...")
loader = WebBaseLoader("https://python.langchain.com/docs/")
docs = loader.load()

print("Splitting documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print("Creating embeddings locally with SentenceTransformers...")
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("Creating FAISS vector store...")
vectorstore = FAISS.from_documents(chunks, embedding)

print("Setting up ChatOpenAI model...")
llm = ChatOpenAI(temperature=0)  # Uses the key set above

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True,
)

print("Chatbot ready! Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = qa_chain({"question": query})
    print("\n🤖 Bot:", result["answer"])
    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'Unknown')}")
    print("\n---\n")

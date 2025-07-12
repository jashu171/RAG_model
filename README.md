# RAG_model

# 🤖 RAG CSV Chatbot using LangChain & Gemini 2.0 Flash

This project is an AI-powered **Retrieval-Augmented Generation (RAG)** chatbot that lets users upload **CSV files** and ask intelligent, structured questions about the data. It's built using **LangChain**, **ChromaDB**, and **Gemini 2.0 Flash** from **Google Generative AI**.

---

## ✨ Key Features

- 📁 Upload a CSV file and parse structured information.
- 🔍 Retrieve relevant document chunks using **FastEmbed + ChromaDB**.
- 💬 Use **Gemini 2.0 Flash** LLM to answer user queries with context.
- 🧠 Custom prompt engineering for **store analytics, customer data, and performance**.
- 🔒 Secure API key usage via Google Colab's secret manager.
- 📦 Modular design with ingestion, retrieval, and response components.

---

## 🧱 Tech Stack

| Component         | Library / Tool                        |
|------------------|----------------------------------------|
| Language Model    | Gemini 2.0 Flash (Google GenAI)       |
| Framework         | LangChain                             |
| Vector Store      | ChromaDB                              |
| Embeddings        | FastEmbed                             |
| Data Loader       | LangChain CSVLoader                   |
| Platform          | Google Colab or Local Python          |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-csv-chatbot.git
cd rag-csv-chatbot
```

### 2. Install Required Libraries

In your Python environment or Google Colab:

```python
!pip install langchain-community
!pip install langchain-google-genai
!pip install fastembed
!pip install chromadb
!pip install -U langchain-chroma
!pip install pypdf
```

---

## 🔐 API Key Setup (Gemini)

In **Google Colab**, run the following to securely store your Gemini API key:

```python
from google.colab import userdata
userdata.set_secret("GOOGLE_API_KEY", "your-api-key-here")
```

---

## 📁 Upload CSV File

Use any structured CSV file. Example: `Enhanced_Store_Data_10_Stores.csv`

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("/content/Enhanced_Store_Data_10_Stores.csv")
```

---

## 📦 Ingest Data into Vector Store

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest():
    loader = CSVLoader("/content/Enhanced_Store_Data_10_Stores.csv")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} documents into {len(chunks)} chunks")

    embedding = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./sql_chroma_db"
    )
```

---

## 🔧 RAG Chain with Gemini 2.0 Flash

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def rag_chain():
    from google.colab import userdata
    google_api_key = userdata.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Please set the GOOGLE_API_KEY in Colab secrets")

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

    prompt = PromptTemplate.from_template("""
<s> [Instructions] You are a helpful assistant with expertise in analyzing store data.
Based only on the following context from a CSV file, provide a detailed and structured answer to the user's question.
Extract and present key information such as Store ID, Date, Total Sales, POS Transactions and Value, Online Transactions and Value, People Counting, Vehicles Parked, Footfall Peak Hour, Average Dwell Time, Employee Count, Customer Satisfaction Score, and any detected incidents (Gun Detection, Theft Detection, Face Recognition Alerts) relevant to the query.
Present this information clearly, preferably using bullet points or a summary paragraph.
If the answer is not found in the context, say so clearly.
[/Instructions]</s>
[Instructions] Question: {input}
context: {context}
Answer: [/Instructions]</s>
""")

    embedding = FastEmbedEmbeddings()
    vectorstore = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    return chain
```

---

## 💬 Ask a Question

```python
def ask(query: str):
    chain = rag_chain()
    results = chain.invoke({"input": query})

    print("Answer:\n", results['answer'])
    for doc in results["context"]:
        print("Source:", doc.metadata['source'])
```

Example usage:

```python
ask("What is the total sales value for Store ID 1002 on March 10?")
```

---

## 📂 Directory Structure

```
rag-csv-chatbot/
├── sql_chroma_db/               # Persisted vector DB
├── Enhanced_Store_Data.csv      # Example data file
├── ingest.py                    # Data ingestion script
├── rag_chain.py                 # RAG chain builder
├── app.py                       # Optional Flask UI
└── README.md                    # This file
```

---

## 🧠 How It Works

1. **Data Ingestion**: Reads and chunks your uploaded CSV.
2. **Embedding**: Each chunk is embedded using FastEmbed.
3. **Vector Store**: Stores embeddings in ChromaDB.
4. **Retrieval**: Finds similar chunks based on user query.
5. **LLM Response**: Gemini 2.0 Flash generates detailed response using prompt + context.

---

## 📈 Future Enhancements

- ✅ Flask/Streamlit UI for local deployment
- 🔄 Multi-file support (CSV, PDF, XLSX)
- 🧩 Auto file-type detection
- 📊 Summarization and charts generation
- 🔐 User login/session context retention

---

## 📄 License

MIT License © 2025 jashu 

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have ideas to improve the chatbot.

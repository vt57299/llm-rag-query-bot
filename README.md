# 🔍 RAG-Ollama-Assistant

A lightweight and modular Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ChromaDB**, and **Ollama**, enabling LLMs to answer questions based on your custom document data.

---

## 🚀 Overview

This project implements a simple but powerful RAG pipeline that allows you to query local or domain-specific documents using a Large Language Model (LLM). The system retrieves relevant documents from a vector store (ChromaDB), constructs a prompt with contextual information, and invokes the **Mistral model** (via Ollama) to generate answers.

---

## 🧠 Features

- 🔎 **Context-aware querying** with semantic search
- 💬 **Ollama** integration for local LLM inference
- 🧱 Built on **LangChain** for modularity
- 🧠 Uses **ChromaDB** for fast document retrieval
- 🗃️ Easily extendable to support custom datasets or other LLMs

---

## 📁 Project Structure

```
.
├── embedding_function.py       # Embedding logic for documents
├── populate_database.py              # Script to load and embed data into ChromaDB
├── query_data.py               # Main RAG pipeline for querying
├── test_rag.py                 # Evaluation script to test RAG accuracy using Mistral LLM
├── chroma/                     # Persistent vector store directory
├── requirements.txt            # Project dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Component        | Description                              |
|------------------|------------------------------------------|
| **LangChain**    | Orchestrates the embedding & querying pipeline |
| **ChromaDB**     | Local vector store for storing embeddings |
| **Ollama**       | Interface for local LLMs (Mistral, etc.)  |
| **Mistral**      | The local LLM used for generating responses |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vt57299/llm-rag-query-bot.git
cd llm-rag-query-bot
```

### 2. Create Virtual Environment

```bash
python -m venv rag-venv
source rag-venv/bin/activate  # On Windows: rag-venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Start Ollama and Pull Mistral Model

```bash
ollama run mistral
```

---

## 📌 Usage

### 1. Ingest Documents

Place your `.pdf`, or other supported files in a directory and update `populate_database.py` accordingly:

```bash
python populate_database.py
```

### 2. Query the Assistant

```bash
python query_data.py "How to get out of jail in monopoly?"
```

---

#### 💬 Example Output:

```text
Response: You can get out of Jail in Monopoly by throwing doubles on any of your next three turns,
using the "Get Out of Jail Free" card if you have it, purchasing and playing the "Get Out of Jail Free"
card from another player, or paying a fine of $50 before you roll the dice on either of your next two turns.
If you do not throw doubles by your third turn, you must pay the $50 fine and then get out of Jail.

Sources: [
  'data\\monopoly.pdf:4:1',
  'data\\monopoly.pdf:4:2',
  'data\\monopoly.pdf:4:0',
  'data\\monopoly.pdf:2:2',
  'data\\monopoly.pdf:1:1'
]
```

---

## 📌 Notes

- Ensure that **Ollama** is running before querying.
- ChromaDB is persisted locally in the `chroma/` directory.
- You can swap the model in `OllamaLLM(model="mistral")` to another one available in Ollama.

---


## 👤 Author

**Vivek Thakur**  
Generative AI Engineer | Python Backend | LLMs + RAG  
🔗 [GitHub](https://github.com/vt57299) • [LinkedIn](https://linkedin.com/in/vivek-thakur-7079aa17b)

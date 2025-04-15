from _1_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import argparse
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma" 

PROMPT_TEMPLATE = """
You are an expert assistant. Answer the question **only** using the provided context.
**"if the answer is not in the context, say **"I don't know based on the provided information"**

### **Context:**
{context}

### **Question:**
{question}

### **Answer:**
"""

def query_rag(query_text):
    """
    Queries the Chroma vector database and generates an answer using the Ollama model.

    Args:
        query_text (str): The user's question.

    Steps:
        1. Load the Chroma database with the correct embedding function.
        2. Perform a similarity search to retrieve relevant documents.
        3. Format the retrieved documents into a structured prompt.
        4. Pass the prompt to Ollama model to generate a response.
        5. Return and print the response along with the document source.

    Returns:
        str: The generated response from the model.
    """

    # Load the embedding function
    embedding_function = get_embedding_function()

    # Initialize database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # search for similar documents
    results = db.similarity_search_with_score(query=query_text, k = 5)

    # Extract context from retrieved documents
    context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query_text)

    model = OllamaLLM(model = "mistral")
    response_text = model.invoke(prompt)

    # Extract document sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return response_text

def main():
    """
    Main function to handle command-line input and perform a RAG-based query.

    - Parses a query text from the command line.
    - Calls `query_rag` to retrive relevant documents and generate an answer.
    """
    parser = argparse.ArgumentParser(description="Query the RAG-based system.")
    parser.add_argument("query_text", type=str, help="Text to query.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

if __name__=="__main__":
    main()
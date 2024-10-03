from flask import Flask, render_template, request, jsonify
import os
import time
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Initialize model and ChromaDB
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
llama_model = OllamaLLM(model="llama3")
template = """
You are a friendly and personalized assistant that helps recommend products from Amazon based on the user's specific needs!
Focus only on the product the user asks about (e.g., if the user asks about a TV, only discuss TVs, and avoid other products like laptops or monitors).

Follow this step-by-step flow for interaction:
1. Greet the user warmly.
2. Ask detailed questions to fully understand their preferences. For example:
    - 'Do you have a brand preference?'
    - 'What specific features are important to you (e.g., screen size, resolution)?'
    - 'What is your budget range?'
3. Do **NOT** recommend any products until the user's preferences are clear.
4. Recommend a product **only** after gathering enough details and personalize the response. Provide a list of the best-matching products (if there are multiple), comparing key features.
5. If the user specifies a preferred product, provide detailed information and a purchase link (if available).
6. If the user asks for another product category, repeat the process.

Here is the context and conversation history: {context}

User Question: {question}

Answer:
"""
prompt_template = ChatPromptTemplate.from_template(template)
chain = prompt_template | llama_model

PDF_FOLDER_PATH = r'C:\Users\spriy\OneDrive\Desktop\Anish Python Projects 1\Amazon_Recommender\k'  # Change this to your PDF folder path
collection_name = "conversation_history"

def reset_collection(db, collection_name):
    """Delete the collection if it exists and create a new one."""
    try:
        db.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")
    except ValueError:
        print(f"Collection '{collection_name}' does not exist, so no need to delete.")
    
    collection = db.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created.")
    return collection

db = chromadb.Client()
collection = reset_collection(db, collection_name)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    import pdfplumber
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def add_text_to_db(text, metadata):
    """Add text and metadata to Chroma DB."""
    embeddings = model.encode([text], convert_to_numpy=True)
    doc_id = metadata.get('file_path', str(hash(text)))
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=embeddings,
        metadatas=[metadata]
    )
    print(f"Document added with id {doc_id}. Total documents in collection: {len(collection.get()['documents'])}")

def add_conversation_to_db(message, role):
    """Embed and store a message in ChromaDB with metadata (role: 'user' or 'AI')."""
    embedding = model.encode([message], convert_to_numpy=True)
    metadata = {"role": role, "message": message}
    collection.add(
        ids=[str(hash(message))],
        documents=[message],
        embeddings=embedding,
        metadatas=[metadata]
    )
    print(f"Conversation added to DB: {role} -> {message[:50]}...")  

def retrieve_relevant_history(user_input, n_results=5):
    """Retrieve relevant conversation history and document information from Chroma DB based on the user's input."""
    query_embedding = model.encode([user_input], convert_to_numpy=True)
    query_args = {"query_embeddings": query_embedding, "n_results": n_results}
    results = collection.query(**query_args)
    relevant_context = ""
    if len(results['documents']) == 0:
        print(f"No relevant documents found for query: {user_input}")
        return ""
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        role = meta.get('role', 'unknown role')
        relevant_context += f"{role}: {doc}\n"
    print(f"Retrieved {len(results['documents'][0])} relevant history for user input: {user_input}")
    return relevant_context

def get_relevant_documents(user_input, top_k=3):
    """Retrieve relevant documents from Chroma DB based on user input and summarize top-k results."""
    query_embedding = model.encode([user_input], convert_to_numpy=True)
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    if results['documents']:
        summarized_docs = [doc for doc_group in results['documents'] for doc in doc_group]
        return "\n\n".join(summarized_docs)
    print(f"No relevant documents found for user input: {user_input}")
    return ""

def handle_user_input(user_input):
    """Process user input, update conversation history, and generate a response."""
    greetings = ["hi", "hello", "hey", "greetings", "what's up"]
    if user_input.lower() in greetings:
        model_response = "Hello! How can I assist you today? Are you looking for any specific products?"
        print(model_response)
        return model_response

    relevant_context = retrieve_relevant_history(user_input)
    relevant_docs = get_relevant_documents(user_input, top_k=3)
    context = f"{relevant_context}\n\nContext:\n{relevant_docs}\n"
    prompt_data = {"context": context, "question": user_input}
    result = chain.invoke(prompt_data)
    model_response = result

    add_conversation_to_db(user_input, role="user")
    add_conversation_to_db(model_response, role="AI")

    history_length = len(collection.get()['documents'])
    print(f"Total conversation history length: {history_length}")

    return model_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = handle_user_input(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

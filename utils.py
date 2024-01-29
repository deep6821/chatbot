import faiss
import fitz
import hashlib
import json
import openai
import numpy as np
import spacy
from sentence_splitter import split_text_into_sentences
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import uuid


# Initialize OpenAI client
client = openai.OpenAI(api_key='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')
# Path to Faiss index
faiss_index_path = "C:\\office\\chatbot\\index_store\\faiss_index.index"
# ---
nlp = spacy.load("en_core_web_md")
# Open AI's token splitting library
tiktokenInstance = tiktoken.encoding_for_model("gpt-3.5-turbo")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()

    return text

def get_actual_page_number(pdf_file_path, target_text):
    doc = fitz.open(pdf_file_path)
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        if target_text in text:
            # Page numbers are 1-based
            return page_num + 1

    # Return None if the target text is not found in any page
    return None

def tokenize(text):
    return tiktokenInstance.encode(text)

def detokenize(tokens):
    return tiktokenInstance.decode(tokens)

def chunks_to_list(lst, n):
    new_lst = []
    for i in range(0, len(lst), n):
        new_lst.append(lst[i: i + n])
    return new_lst

def token_split(text, max_tokens=4060):
    tokens = tokenize(text)
    return [detokenize(c) for c in chunks_to_list(tokens, max_tokens)]

# def token_and_sentence_split(text, max_tokens=512, max_sentence_tokens=512):
def token_and_sentence_split(text, max_tokens=4096, max_sentence_tokens=512):
    raw_sentences = split_text_into_sentences(text, "en")
    chunk_token_len = 0
    chunk_texts = []
    splitted = []
    sentences = []

    for sentence in raw_sentences:
        if len(tokenize(sentence)) > max_sentence_tokens:
            sentences += token_split(sentence, max_tokens)
        else:
            sentences.append(sentence)

    for sentence in sentences:
        token_len = len(tokenize(sentence))
        if chunk_token_len + token_len > max_tokens:
            if chunk_texts:
                splitted.append(" ".join(chunk_texts))
            chunk_texts = []
            chunk_token_len = 0

        chunk_texts.append(sentence)
        chunk_token_len += token_len

    if chunk_texts:
        splitted.append(" ".join(chunk_texts))

    splitted = [s for s in splitted if s]
    return splitted

def generate_openai_embedding(text, max_tokens=4096):
    print("Here ----------------------------------")
    client = openai.OpenAI(api_key='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')
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text],
    )
    
    embedding_list = response.data[0].embedding
    return embedding_list

def save_to_faiss_index(data, pdf_file_path, faiss_index_path):
    document_ids, page_numbers, texts = [], [], []
    # Get the entire text from each chunk
    for chunk in data:
        document_ids.append(chunk["document_id"])
        page_numbers.append(chunk["page_number"])
        texts.append(chunk["text"])

    # Generate OpenAI embeddings for each text
    embeddings = [generate_openai_embedding(text) for text in texts]

    # Determine the maximum length among all embeddings
    max_length = len(embeddings[0])

    # Build Faiss index with ID mapping
    index = faiss.IndexIDMap(faiss.IndexFlatL2(max_length))

    # Add valid sentence arrays to the index
    for i, (embedding, page_number) in enumerate(zip(embeddings, page_numbers)):
        embedding_array = np.array(embedding, dtype=np.float32)
        index.add_with_ids(embedding_array.reshape(1, -1), np.array([page_number]))

    # Save the Faiss index with metadata to a file
    faiss.write_index(index, faiss_index_path)

    # Save metadata including pdf_file_path for each chunk
    metadata_list = [{"pdf_file_path": pdf_file_path, "page_number": chunk["page_number"]} for chunk in data]
    with open(faiss_index_path + "_metadata.json", "w") as metadata_file:
        json.dump(metadata_list, metadata_file)

    # # Add valid sentence arrays to the index
    # for i, (embedding, page_number, document_id) in enumerate(zip(embeddings, page_numbers, document_ids)):
    #     combined_id = f"{page_number}_{document_id}"
    #     embedding_array = np.array(embedding, dtype=np.float32)
    #     index.add_with_ids(embedding_array.reshape(1, -1), np.array([combined_id]))

    # # Save the Faiss index with metadata to a file
    # faiss.write_index(index, faiss_index_path)

    # # Save metadata including pdf_file_path and document_idx for each chunk
    # metadata_list = [{"pdf_file_path": pdf_file_path, "page_number": chunk["page_number"], "document_id": chunk["document_id"]} for chunk in data]
    # with open(faiss_index_path + "_metadata.json", "w") as metadata_file:
    #     json.dump(metadata_list, metadata_file)

def hash_document_id(document_id, max_value=1000):
    # Hash the document ID using a cryptographic hash function (SHA-256)
    hash_object = hashlib.sha256(document_id.encode())
    hash_digest = hash_object.digest()
    # Convert the hash digest to a numerical value by interpreting it as an integer
    hash_int = int.from_bytes(hash_digest, byteorder='big')
    truncated_hash_int = hash_int % max_value
    return truncated_hash_int

def process_and_save_pdf(pdf_file_path, faiss_index_path):
    doc = fitz.open(pdf_file_path)
    split_pdf_data = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_text = page.get_text()
        actual_page_number = page_num + 1

        # Split the page text into chunks
        chunks = token_and_sentence_split(page_text)

        for chunk in chunks:
            document_id = hash_document_id(str(uuid.uuid4()))
            split_pdf_data.append({"text": chunk, "page_number": actual_page_number, "document_id": document_id})

    # Save OpenAI embedding and actual page numbers to Faiss index
    save_to_faiss_index(split_pdf_data, pdf_file_path, faiss_index_path)

# Function to identify if a text is a question
def identify_question(text):
    return text.strip().endswith('?')

def generate_embedding(text, max_tokens=4096):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text],
    )
    
    embedding_list = response.data[0].embedding
    return embedding_list

# Function to load metadata from a file
def load_metadata_from_file(metadata_file_path):
    with open(metadata_file_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata

# Function to get text from a specific page of a PDF file
def get_text_from_page(pdf_file_path, page_number):
    doc = fitz.open(pdf_file_path)
    if 1 <= page_number <= doc.page_count:
        page = doc[page_number - 1]
        text = page.get_text()
        return text
    else:
        return None

# Function to search the knowledge base for a question
def search_in_knowledge_base(question, faiss_index_path):
    # Load the Faiss index with metadata
    index = faiss.read_index(faiss_index_path)

    # Generate an embedding for the question
    question_embedding = generate_embedding(question)

    # Perform the search in the vector database. You can adjust the number of nearest neighbors to retrieve
    k = 5
    distances, indices = index.search(np.array([question_embedding]), k)

    # Extract metadata (page number, pdf file path) for the retrieved vectors
    metadata = load_metadata_from_file(faiss_index_path + "_metadata.json")

    return metadata, indices, distances

def calculate_similarity(reference_text, generated_text):
    # Use semantic similarity as a metric
    reference_embedding = nlp(reference_text).vector
    generated_embedding = nlp(generated_text).vector

    similarity_score = cosine_similarity([reference_embedding], [generated_embedding])[0][0]
    return similarity_score

# Function to find the best answer based on retrieved metadata and question
def find_best_answer(metadata_list, question, max_tokens=4096):
    highest_similarity = 0
    best_response = None

    for metadata in metadata_list:
        page_number = metadata["page_number"] - 1
        pdf_file_path = metadata["pdf_file_path"]
        pdf_text_data = get_text_from_page(pdf_file_path, page_number)

        max_text_length = max_tokens - len(f"Question: {question} Page: {page_number} and PDF: {pdf_file_path} ")
        truncated_text = pdf_text_data[:max_text_length]

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question: {question} Page: {page_number} and PDF: {pdf_file_path} {truncated_text}"}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=500,
            temperature=0,
        )
        generated_answer = response.choices[0].message.content

        similarity = calculate_similarity(truncated_text, generated_answer)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_response = {
                "Question": question,
                "Answer": generated_answer,
                "PDF_Link_with_Page": f"{pdf_file_path} and Page Number {page_number}"
            }

    # Check if the highest similarity meets a threshold, adjust as needed
    similarity_threshold = 0.5
    if highest_similarity >= similarity_threshold:
        return best_response
    else:
        return {
            "Question": question,
            "Answer": best_response,
            "PDF_Link_with_Page": "No specific link provided"
        }

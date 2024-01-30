import faiss
import fitz  # PyMuPDF
import hashlib
import json
import openai
import numpy as np
import pandas as pd
from sentence_splitter import split_text_into_sentences
import tiktoken
import uuid


# Update with your desired path
faiss_index_path = "C:\\office\\chatbot\\index_store\\faiss_index.index"
# faiss_index_path = "C:\\office\\chatbot\\index_store\\test_faiss_index.index"
pdf_file_path = "C:\\Users\\rohitpandey02\\Downloads\\HAI_AI-Index-Report_2023.pdf"

# Open AI's token splitting library
# tiktokenInstance = tiktoken.encoding_for_model("text-davinci-003")
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
def token_and_sentence_split(text, max_tokens=4096, max_sentence_tokens=1024):
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
    client = openai.OpenAI(api_key='YOUR-API-KEY')
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
        # document_ids.append(chunk["document_id"])
        page_numbers.append(chunk["page_number"])
        texts.append(chunk["text"])

    print("document_ids", len(document_ids))
    print("page_numbers", len(page_numbers))
    print("texts", len(texts))

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
    metadata_list = [{"pdf_file_path": pdf_file_path, "chunk_text": chunk["text"], "page_number": chunk["page_number"]} for chunk in data]
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
            # document_id = hash_document_id(str(uuid.uuid4()))
            # split_pdf_data.append({"text": chunk, "page_number": actual_page_number, "document_id": document_id})

            split_pdf_data.append({"text": chunk, "page_number": actual_page_number})

    # Save OpenAI embedding and actual page numbers to Faiss index
    save_to_faiss_index(split_pdf_data, pdf_file_path, faiss_index_path)

def process_and_save_csv(csv_file_path, faiss_index_path):
    # Read CSV file using pandas
    df = pd.read_csv(csv_file_path)

    # Assuming all columns are numeric and represent vectors
    vectors = df.to_numpy(dtype='float32')

    # Extract page numbers from the "page_number" column
    page_numbers = df["page_number"].tolist()

    # Save CSV data and page numbers to Faiss index
    save_to_faiss_index(vectors, [{"page_number": page} for page in page_numbers], faiss_index_path)

if __name__ == "__main__":
    # Process and save PDF data to Faiss index
    process_and_save_pdf(pdf_file_path, faiss_index_path)

    # Process and save CSV data to Faiss index
    # process_and_save_csv(csv_file_path, faiss_index_path)

import faiss
import fitz
import json
import openai
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import subprocess
import tempfile


nlp = spacy.load("en_core_web_md")
client = openai.OpenAI(api_key='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')
faiss_index_path = "C:\\office\\chatbot\\index_store\\faiss_index.index"
# faiss_index_path = "C:\\office\\chatbot\\index_store\\test_faiss_index.index"


def convert_speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    audio_data  = audio.get_wav_data()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        temp_wav_file.write(audio_data)
        temp_wav_file_path = temp_wav_file.name
    
    temp_flac_file_path = temp_wav_file_path.replace(".wav", ".flac")
    subprocess.run(["ffmpeg", "-i", temp_wav_file_path, temp_flac_file_path])
    
    with open(temp_flac_file_path, "rb") as flac_file:
        response = client.audio.transcriptions.create(
            file=flac_file,
            model="whisper-1",
            language="en",
            response_format="text",
            temperature=0.2,
        )
        return response
    
def identify_question(text):
    return text.strip().endswith('?')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()

    return text

def get_text_from_page(pdf_file_path, page_number):
    doc = fitz.open(pdf_file_path)
    
    if 1 <= page_number <= doc.page_count:
        page = doc[page_number - 1]
        text = page.get_text()
        return text
    else:
        return None
    
def generate_embedding(text, max_tokens=4096):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text],
    )
    
    embedding_list = response.data[0].embedding
    return embedding_list

def load_metadata_from_file(metadata_file_path):
    with open(metadata_file_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata

def search_in_knowledge_base(question, faiss_index_path):
    # Load the Faiss index with metadata
    index = faiss.read_index(faiss_index_path)

    # Generate an embedding for the question
    question_embedding = generate_embedding(question)

    # Perform the search in the vector database. You can adjust the number of nearest neighbors to retrieve
    k = 4
    distances, indices = index.search(np.array([question_embedding]), k)

    # Extract metadata (page number, pdf file path) for the retrieved vectors
    metadata = load_metadata_from_file(faiss_index_path + "_metadata.json")

    # Return the retrieved metadata or embeddings as needed
    return metadata, indices, distances

def calculate_similarity(reference_text, generated_text):
    # Use semantic similarity as a metric
    reference_embedding = nlp(reference_text).vector
    generated_embedding = nlp(generated_text).vector

    similarity_score = cosine_similarity([reference_embedding], [generated_embedding])[0][0]
    return similarity_score

def find_best_answer(metadata_list, question, max_tokens=4096):
    best_response = None
    default_answer = "I'm sorry, I couldn't find the answer to your question in the provided context."
    highest_similarity = 0

    prompt = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question as truthfully as possible using the provided context and if the answer is not contained within the text below, say 'I am sorry, I couldn't find the answer to your question in the provided context.'"},
        {"role": "user", "content": ""}
    ]

    for metadata in metadata_list:
        pdf_file_path = metadata["pdf_file_path"]
        chunk_text = metadata["chunk_text"]
        page_number = metadata["page_number"]

        # max_text_length = max_tokens - len(f"Question: {question} Page: {page_number} and PDF: {pdf_file_path} ")
        # chunk_text = chunk_text[:max_text_length]

        # Update the user content in the prompt
        prompt[1]["content"] = f"Question: {question} Page: {page_number} and PDF: {pdf_file_path} {chunk_text}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=500,
            temperature=0,
        )
        generated_answer = response.choices[0].message.content

        similarity = calculate_similarity(chunk_text, generated_answer)

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
        if  "I am sorry" in best_response["Answer"]:
            best_response["Answer"] = default_answer
            best_response["PDF_Link_with_Page"] = "No specific link provided"
            return best_response
        return best_response
    else:
        return {
            "Question": question,
            "Answer": default_answer,
            "PDF_Link_with_Page": "No specific link provided"
        }

if __name__ == "__main__":
    print("Welcome to the Knowledge Base bot. Type 'exit' or 'quit' to stop")
    conversation_data = []
    while True:
        # customer_question = "Who launched ReClor?"
        # customer_question = convert_speech_to_text()
        customer_question = input("\nEnter your question: ")
        if customer_question and (customer_question.lower() == "exit" or customer_question.lower() == "quit"):
            break

        if identify_question(customer_question):       
            # Call the function to search in the vector database
            metadata, indices, distances = search_in_knowledge_base(customer_question, faiss_index_path)
      
            answer = find_best_answer([metadata[i-1] for i in indices[0]], customer_question)

            # Display the response
            print("\nOpenAI Chat Completion Answer:", answer)

            conversation_data.append(answer)
    
    print("\nconversation_data: ", conversation_data)

import faiss
import fitz
import json
import openai
import numpy as np
import speech_recognition as sr
import subprocess
import tempfile


client = openai.OpenAI(api_key='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')
faiss_index_path = "C:\\office\\chatbot\\index_store\\faiss_index.index"

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
    # Implement code to identify question from the text
    # Example: return text.split('?')[0]
    # return text
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
    k = 5
    distances, indices = index.search(np.array([question_embedding]), k)

    # Extract metadata (page number, pdf file path) for the retrieved vectors
    metadata = load_metadata_from_file(faiss_index_path + "_metadata.json")

    # Display detailed information for debugging
    # print("Distances:", distances)
    # print("Indices:", indices)
    # print("\n")

    # # Display the retrieved metadata
    # for i in range(k):
    #     page_number = metadata[indices[0][i]]["page_number"]
    #     pdf_file_path = metadata[indices[0][i]]["pdf_file_path"]
    #     distance = distances[0][i]  # Distance for the ith result
    #     # print(f"Result {i + 1}: Page {page_number - 1}, PDF: {pdf_file_path}, Distance: {distance}")

    # Return the retrieved metadata or embeddings as needed
    return metadata, indices, distances

def find_best_answer(metadata, question, max_tokens=4096):
    page_number = metadata["page_number"] -1
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
        temperature=0.7,
    )
    gpt_answer = response.choices[0].message.content

    return {
        "Question": question,
        "Answer": gpt_answer,
        "PDF_Link_with_Page": f"{pdf_file_path} and Page Number {page_number}"
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
            result_metadata = metadata[indices[0][0]]

            # Use chat completion to generate a response
            answer = find_best_answer(result_metadata, customer_question)

            # Open document and display answer
            print("\nOpenAI Chat Completion Answer:", answer)

            conversation_data.append(answer)

    print("Conversation Data: ", conversation_data)
    
import numpy as np
import openai
import pandas as pd
import PyPDF2
import speech_recognition as sr



class KnowledgeBaseBot:
    def __init__(self):
        self.client = openai.OpenAI(api_key='YOUR API KEY')
        # Store conversation data
        self.conversation_data = []
        self.knowledge_base = {
            "what is your name": "I am a knowledge base bot.",
            "how does photosynthesis work": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigments.",
        }
        self.pdf_path = "c:\\Users\\rohitpandey02\\Downloads\\Nagarro_Project_Overview.pdf"
    
    def convert_speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand. Please repeat your question.")
            return None

    def process_customer_question(self, customer_question):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Customer: {customer_question}"}
            ]
        )
        print("Response", response)
        return response.choices[0].message.content

    def search_kb(self, customer_question):
        # Simplified KB search logic
        if customer_question.lower() in self.knowledge_base:
            return self.knowledge_base[customer_question.lower()]
        else:
            # Return None if not found in KB
            return None
    
    def search_pdf(self, customer_question, max_paragraphs=1):
        with open(self.pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extracted_text = ""

            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text = page.extract_text().lower()
                text = ' '.join(text.split())

                if customer_question.lower() in text:
                    # Extract the text from the location of the customer question to the fist paragraph of the document
                    start_index = text.find(customer_question.lower())
                    extracted_text = text[start_index:].strip()

                    # Limit the extracted text to a certain number of paragraphs
                    # paragraphs = extracted_text.split('\n\n')[:max_paragraphs]
                    paragraphs = extracted_text.split('.')[:max_paragraphs]
                    extracted_text = '\n\n'.join(paragraphs)
                    # Page numbers start from 1
                    return extracted_text, page_number + 1

        return None, None

    def open_document(self, page_number):
        print(f"Document Link: {self.pdf_path} and user question found in page number: {page_number}")
        return f"{self.pdf_path} and {page_number}"
    
    def update_conversation_data(self, user_message, bot_message, document_info=None):
        conversation_entry = {
            "Human to human conversation": user_message,
            "Chatbot for answering questions": bot_message,
            "Document link with page number": document_info
        }
        self.conversation_data.append(conversation_entry)

    def display_answer(self, answer):
        print(f"Answer {answer}")


if __name__ == "__main__":
    kb_bot = KnowledgeBaseBot()
    print("Welcome to the Knowledge Base bot. Type 'exit' or 'quit' to stop")
    while True:
        print("What is your question?")
        # customer_question = kb_bot.convert_speech_to_text()
        customer_question = "Hello how are you"
        if customer_question and (customer_question.lower() == "exit" or customer_question.lower() == "quit"):
            break
                
        if customer_question is not None:
            # Search in the KB first
            kb_answer = kb_bot.search_kb(customer_question)
            if kb_answer:
                kb_bot.display_answer(kb_answer)
                kb_bot.update_conversation_data(customer_question, kb_answer)
            else:
                # If not found in KB, search in the PDF
                extracted_text, pdf_page_number = kb_bot.search_pdf(customer_question)
                if pdf_page_number is not None:
                    kb_bot.display_answer(extracted_text)
                    document_info = kb_bot.open_document(pdf_page_number)
                    kb_bot.update_conversation_data(customer_question, extracted_text, document_info)
                else:
                    # If not found in PDF, use NLU to get a response
                    assistant_response = kb_bot.process_customer_question(customer_question)
                    kb_bot.display_answer(assistant_response)
                    kb_bot.update_conversation_data(customer_question, assistant_response)
    
    # Create a Pandas DataFrame from the conversation data
    df = pd.DataFrame(kb_bot.conversation_data)
    print("\nConversation DataFrame:")
    print(df.to_dict(orient='records'))


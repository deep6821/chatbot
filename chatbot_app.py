from flask import Flask, render_template, request, jsonify
import os
import tempfile

from utils import process_and_save_pdf, faiss_index_path, identify_question, search_in_knowledge_base, find_best_answer

app = Flask(__name__)


# Flask application routes
@app.route("/")
def index():
    return render_template("chatbot.html")

# Route for uploading and processing PDF file data
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename.endswith('.pdf'):
        # Save the uploaded file temporarily
        temp_pdf_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_pdf_file_path = temp_pdf_file.name
        file.save(temp_pdf_file_path)

        try:
            # Process and save PDF data to Faiss index
            process_and_save_pdf(temp_pdf_file_path, faiss_index_path)
            return jsonify({"success": "PDF file uploaded and processed successfully"})
        finally:
            # Remove the temporary PDF file
            os.unlink(temp_pdf_file_path)

    return jsonify({"error": "Invalid file format. Please upload a PDF file"})

@app.route("/process_text", methods=["POST"])
def process_text():
    text = request.form.get("text")
    if identify_question(text):
        metadata, indices, distances = search_in_knowledge_base(text, faiss_index_path)
        result_metadata = metadata[indices[0][0]]
        answer = find_best_answer(result_metadata, text)
        return jsonify(answer)
    else:
        return jsonify({"error": "Input is not a question"})
    

if __name__ == "__main__":
    app.run(debug=True)

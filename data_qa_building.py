import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import json


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


# Function to generate questions and answers
def generate_questions_and_answers(text):
    # Load pipelines
    qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    qa_pipeline = pipeline("question-answering")

    # Split the text into chunks to avoid max length issue
    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]

    qas = []
    for chunk in chunks:
        # Generate questions
        qg_input = f"generate questions: {chunk}"
        questions = qg_pipeline(qg_input)

        # Answer the generated questions
        for q in questions:
            question = q['generated_text']
            answer = qa_pipeline(question=question, context=chunk)
            qas.append({
                "question": question,
                "answer": answer['answer']
            })
    return qas


# Streamlit app
st.title("SQuAD Data Generator from Biomedical Papers")

uploaded_file = st.file_uploader("Upload a PDF paper", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully")

    # Extract text from PDF
    paper_text = extract_text_from_pdf("temp.pdf")
    st.write("Extracted text from the paper:")
    st.write(paper_text[:2000] + "...")  # Display first 2000 characters

    if st.button("Generate Questions and Answers"):
        with st.spinner("Generating questions and answers..."):
            qas = generate_questions_and_answers(paper_text)

            # Save to JSON file
            output_path = "qas_output.json"
            with open(output_path, "w") as f:
                json.dump(qas, f, indent=4)

            st.success("Questions and answers generated successfully")

            # Display QAs
            st.write(qas)

            # Download link for JSON file
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download JSON",
                    data=f,
                    file_name="qas_output.json",
                    mime="application/json"
                )

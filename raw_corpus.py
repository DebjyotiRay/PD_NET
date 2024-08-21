import fitz  # PyMuPDF
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def chunk_text(text, max_length):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

def answer_question(question, text, tokenizer, model):
    max_length = 512
    answers = []

    for chunk in chunk_text(text, max_length - len(tokenizer.encode(question))):
        inputs = tokenizer.encode_plus(question, chunk, add_special_tokens=True, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        if answer:
            answers.append(answer)

    # Return the most common answer
    return max(set(answers), key=answers.count) if answers else "No answer found"

def main(pdf_path, question):
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = BertForQuestionAnswering.from_pretrained("dmis-lab/biobert-v1.1")

    text = extract_text_from_pdf(pdf_path)

    answer = answer_question(question, text, tokenizer, model)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    pdf_path = r"C:\Users\DELL\DEV\GRAPH\KG+LLM.pdf"
    question = "What are the pros of LLMs mentioned in the paper?"

    main(pdf_path, question)

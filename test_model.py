from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load model and tokenizer
model_path = "./resume-summarizer-model"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Sample resume text (replace this with real input later)
resume_text = """
Experienced software engineer with a strong background in Python, machine learning, and cloud technologies.
Worked on various end-to-end ML pipelines in production. Seeking to apply my skills at a company focused on AI innovation.
"""

# Encode input
inputs = tokenizer([resume_text], max_length=1024, return_tensors="pt", truncation=True)

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=30,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True,
)

# Decode and print summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:")
print(summary)

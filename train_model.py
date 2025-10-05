

from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

# Load the dataset from the JSON file
dataset = load_dataset("json", data_files="resume_dataset.json")

# Load tokenizer and model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def preprocess(example):
    inputs = tokenizer(example["resume_text"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(example["summary"], truncation=True, padding="max_length", max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset["train"].map(preprocess)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Start training
trainer.train()
print("Model training complete and saved!")
# Save model
model.save_pretrained("resume-summarizer-model")
tokenizer.save_pretrained("resume-summarizer-model")



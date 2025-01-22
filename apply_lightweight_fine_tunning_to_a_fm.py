# Project: Apply Lightweight Fine-Tuning to a Foundational Model
from peft import (AutoPeftModelForSequenceClassification,
                  LoraConfig,
                  get_peft_model,
                  PeftModel,
                  PeftType,
                  TaskType)
from datasets import (load_dataset)
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer,
                          pipeline,
                          TrainingArguments,
                          Trainer
                          )
from torch.nn.functional import softmax
import evaluate
import random
import torch

# Define the metric variable
metric = evaluate.load("accuracy")
# Define the model to use
model_name = "distilbert-base-uncased"


# Function to tokenize the data
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
    tokenized["labels"] = examples["label"]
    return tokenized


# Function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    print("Predictions and labels:", predictions, labels)
    return metric.compute(predictions=predictions, references=labels)


def evaluate_singleton(text_to_evaluate, model):
    inputs = tokenizer(text_to_evaluate, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1).detach().numpy()

    print("\nDetailed Evaluation on One Sample:")
    print(f"Input Text: {text_to_evaluate[:200]}...\nProbabilities: {probs}\n")


def get_data_subset(data, start_range, stop_range):
    return data[start_range:stop_range]


def evaluate_model(model, tokenizer, data_to_evaluate):
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    predictions = classifier(data_to_evaluate)

    # Print the predictions
    print("\nPredictions on Sample Texts:")
    for text, prediction in zip(data_to_evaluate, predictions):
        print(f"Text: {text[:100]}...\nPrediction: {prediction}\n")


def view_dataset(dataset_to_view, rows=10):
    print("Dataset Preview:")
    for x in range(rows):
        print(dataset_to_view[x])


def train_and_evaluate_model(model_to_train_and_evaluate, tokenizer_to_user, train_ds, test_ds, num_epochs=1, learn_rate=2e-5):
    # Training Arguments
    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        output_dir="./results",
        learning_rate=learn_rate,
    )

    # Trainer
    trainer = Trainer(
        model=model_to_train_and_evaluate,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer_to_user,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    trainer.train()

    # return the evaluation
    return trainer.evaluate()

'''
' Load and process a dataset
'''

'''
' Load a pretrained HF model
'''
# load the imdb dataset
dataset = load_dataset("imdb")
print("Dataset:", dataset)

# Work with about 10% of the downloaded data
train_subset = dataset["train"].train_test_split(train_size=0.01, seed=42)["train"]
test_subset = dataset["test"].train_test_split(test_size=0.01, seed=42)["test"]
print("Training data:", train_subset)
print("Testing data:", test_subset)

# # View some of the data
# view_dataset(train_subset, 10)

# Create the model and tokenizer
# Distilled Bert
# base_model = DistilBertForSequenceClassification.from_pretrained(
#     model_name, num_labels=2
# )
# tokenizer = DistilBertTokenizer.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("Model:", base_model)
print("Tokenizer:", tokenizer)

# Generate the values to extract from the dataset for evaluation
number_rows = len(test_subset)
random_number = random.randint(5, number_rows)
start_pos = random_number-5
stop_pos = random_number-1

'''
' Evaluate the model
'''
evaluate_model(base_model, tokenizer, test_subset['text'][start_pos:stop_pos])

# # Test single entry
# evaluate_singleton("I really, really, really hate it", base_model)
# evaluate_singleton("loved it!", base_model)

# Tokenize the data
# training data
tokenized_train_dataset = train_subset.map(tokenize_function, batched=True)
print("Tokenized training data:", tokenized_train_dataset)
# testing data
tokenized_test_dataset = test_subset.map(tokenize_function, batched=True)
print("Tokenized testing data:", tokenized_test_dataset)

'''
' LoRA
'''
# LoRA model name
lora_model_name = model_name + "-lora"
lora_output_dir = "lora_peft_model"

# Creating a peft config
lora_config = LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=8,
    target_modules=["attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin"],
    lora_dropout=0.1,
    peft_type=PeftType.LORA,
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)

# Create a LoRA model from the pre-trained model
# Converting a transformers model into a peft model
lora_model = get_peft_model(base_model, lora_config)

# Training with a peft model
# Checking the trainable parameters of a peft model
print(lora_model.print_trainable_parameters())

# Train and evaluate the peft model
lora_model_results = train_and_evaluate_model(lora_model, tokenizer, tokenized_train_dataset
                                              , tokenized_test_dataset, 5, float("5e-5"))
print("Fine-tuned model results:", lora_model_results)

# Saving a peft model
lora_model.save_pretrained(lora_output_dir)
print("Saved LoRA Model:", lora_model)

# Load the saved peft model
lora_model_loaded = PeftModel.from_pretrained(base_model, lora_output_dir)
print("Saved Trained LoRA Model:", lora_model_loaded)

# Evaluate the trained LoRA model
print("Evaluating the pre-trained base model...")
evaluate_model(lora_model_loaded, tokenizer, test_subset['text'][start_pos:stop_pos])
# # Test single entry
# evaluate_singleton("I really, really, really hate it", lora_model_loaded)
# evaluate_singleton("loved it!", lora_model_loaded)

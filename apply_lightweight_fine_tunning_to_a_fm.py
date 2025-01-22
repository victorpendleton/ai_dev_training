# Project: Apply Lightweight Fine-Tuning to a Foundational Model
from peft import (AutoPeftModelForSequenceClassification,
                  LoraConfig,
                  get_peft_model,
                  PeftType,
                  TaskType)
from datasets import (load_dataset)
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          pipeline,
                          TrainingArguments,
                          Trainer
                          )
from torch.nn.functional import softmax
import evaluate
import random
import torch

DEBUG = False
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
    print_debug(("Predictions and labels:", predictions, labels), DEBUG)
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


def print_debug(obj, always_print = False):
    if always_print:
        print(obj)


def view_dataset(dataset_to_view, rows=10):
    print("Dataset Preview:")
    for x in range(rows):
        print(dataset_to_view[x])


def train_and_evaluate_model(model_to_train_and_evaluate, tokenizer_to_user, train_ds, test_ds, num_epochs=1, learn_rate=2e-5, train_model=False):
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

    # Train
    if train_model:
        print("Training the model...")
        trainer.train()
    else:
        print("Evaluating the model only...")

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
print_debug(("Dataset:", dataset), True)

# Work with about 10% of the downloaded data
train_subset = dataset["train"].train_test_split(train_size=0.01, seed=42)["train"]
test_subset = dataset["test"].train_test_split(test_size=0.01, seed=42)["test"]
print_debug(("Training data:", train_subset), DEBUG)
print_debug(("Testing data:", test_subset), DEBUG)

# # View some of the data
# view_dataset(train_subset, 10)

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print_debug(("Model:", base_model), DEBUG)
print_debug(("Tokenizer:", tokenizer), DEBUG)

# Generate the values to extract from the dataset for evaluation
number_rows = len(test_subset)
random_number = random.randint(5, number_rows)
start_pos = random_number-5
stop_pos = random_number-1

# # Test single entry
# evaluate_singleton("I really, really, really hate it", base_model)
# evaluate_singleton("loved it!", base_model)

# Tokenize the data
# training data
tokenized_train_dataset = train_subset.map(tokenize_function, batched=True)
print_debug(("Tokenized training data:", tokenized_train_dataset), DEBUG)
# testing data
tokenized_test_dataset = test_subset.map(tokenize_function, batched=True)
print_debug(("Tokenized testing data:", tokenized_test_dataset), DEBUG)

'''
' Evaluate the model
'''
# evaluate_model(base_model, tokenizer, test_subset['text'][start_pos:stop_pos])
print("Evaluating the pre-trained base model...")
base_model_results = train_and_evaluate_model(base_model, tokenizer, tokenized_train_dataset
                                              , tokenized_test_dataset, 1, float("2e-5")
                                              , False)
print_debug(("Base model results:", base_model_results), True)

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
    task_type=TaskType.SEQ_CLS,
    bias="none"
)

# Create a LoRA model from the pre-trained model
# Converting a transformers model into a peft model
lora_model = get_peft_model(base_model, lora_config)

# Training with a peft model
# Checking the trainable parameters of a peft model
lora_model.print_trainable_parameters()

# Train and evaluate the peft model
lora_model_results = train_and_evaluate_model(lora_model, tokenizer, tokenized_train_dataset,
                                              tokenized_test_dataset, 1, float("5e-5"),
                                              True)
print_debug(("Fine-tuned model results:", lora_model_results), True)

# Saving the peft model
lora_model.save_pretrained(lora_output_dir)
print_debug(("Saved LoRA Model:", lora_model), DEBUG)

# Load the saved peft model
lora_model_loaded = AutoPeftModelForSequenceClassification.from_pretrained(lora_output_dir)

print_debug(("Saved Trained LoRA Model:", lora_model_loaded), DEBUG)

# # Evaluate the peft trained LoRA model
peft_lora_model_results = train_and_evaluate_model(lora_model_loaded, tokenizer, tokenized_train_dataset,
                                                   tokenized_test_dataset, 1,
                                                   float("5e-5"), False)
print_debug(("Fine-tuned peft model results:", lora_model_results), True)


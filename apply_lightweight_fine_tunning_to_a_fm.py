# Project: Apply Lightweight Fine-Tuning to a Foundational Model
from transformers import AutoModelForCausalLM
from peft import (AutoPeftModelForSequenceClassification,
                  LoraConfig,
                  get_peft_model,
                  PeftType,
                  TaskType)
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from datasets import (load_dataset)
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer,
                          TrainingArguments,
                          Trainer
                          )
import evaluate
import torch

# Define the metric variable
metric = evaluate.load("accuracy")
# Define the model to use
model_name = "distilbert-base-uncased"
# model_name = "gpt2"


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


# load the imdb dataset
dataset = load_dataset("imdb")
print("Dataset:", dataset)

# Work with about 10% of the downloaded data
train_subset = dataset["train"].train_test_split(train_size=0.01, seed=42)["train"]
test_subset = dataset["test"].train_test_split(test_size=0.01, seed=42)["test"]
print("Training data:", train_subset)
print("Testing data:", test_subset)

# Create the model and tokenizer
# Distilled Bert
base_model = DistilBertForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# # Chat GPT2
# model = AutoModelForCausalLM.from_pretrained("gpt2", num_labels=2)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("Model:", base_model)
print("Tokenizer:", tokenizer)

# Tokenize the data
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# print(tokenized_datasets)
# training data
tokenized_train_dataset = train_subset.map(tokenize_function, batched=True)
print("Tokenized training data:", tokenized_train_dataset)
# testing data
tokenized_test_dataset = test_subset.map(tokenize_function, batched=True)
print("Tokenized testing data:", tokenized_test_dataset)

'''
# Training Arguments
training_args = TrainingArguments(
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    output_dir="./results",
    learning_rate=2e-5,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["test"],
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate
trainer.train()

results = trainer.evaluate()
print(results)
'''

# # Train and evaluate the base model
# base_model_results = train_and_evaluate_model(base_model, tokenizer, tokenized_train_dataset, tokenized_test_dataset, .1)
# print(base_model_results)


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


# LoRA
# LoRA model name
lora_model_name = model_name + "-lora"

# Create a LoRA model from the pre-trained model
# Converting a transformers model into a peft model
lora_model = get_peft_model(base_model, lora_config)

# Training with a peft model
# Checking the trainable parameters of a peft model
lora_model.print_trainable_parameters()

# Saving a peft model
lora_model.save_pretrained(lora_model_name)

# Interface with peft
# Loading a saved peft model
lora_model = AutoPeftModelForSequenceClassification.from_pretrained(lora_model_name)

print("LoRA Model:", lora_model)


# Train and evaluate the models
# base_model_results = train_and_evaluate_model(base_model, tokenizer
# , tokenized_train_dataset, tokenized_test_dataset, float("2e-5"))
lora_model_results = train_and_evaluate_model(lora_model, tokenizer, tokenized_train_dataset
                                              , tokenized_test_dataset, 5, float("5e-5"))
# print("Base model results:", base_model_results)
print("Fine-tuned model results:", lora_model_results)


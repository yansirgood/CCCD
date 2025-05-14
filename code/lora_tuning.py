import os
import json
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback, TrainerControl, TrainerState
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
base_path = "xxxxx"
model_name = "xxxxx"
model_name_path = "xxxxx"
model_folder = os.path.join(base_path, model_name)
model_save_path = os.path.join(model_folder, "model")
result_save_path = os.path.join(model_folder, "result")
log_save_path = os.path.join(model_folder, "logs")
checkpoint_path = os.path.join(model_folder, "checkpoints")
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(result_save_path, exist_ok=True)
os.makedirs(log_save_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)
train_json_path = "xxxxx"
val_json_path = "xxxxx"


tokenizer = AutoTokenizer.from_pretrained(model_name_path)
tokenizer.pad_token = tokenizer.eos_token


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def preprocess_data(data, label_map):
    processed_data = []
    for item in data:
        text = item["instruction"] + item["input"]
        label = label_map[item["answer"]]
        processed_data.append({"text": text, "label": label})
    return processed_data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )


label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
train_data = load_json_data(train_json_path)
val_data = load_json_data(val_json_path)
train_processed = preprocess_data(train_data, label_map)
val_processed = preprocess_data(val_data, label_map)


train_dataset = Dataset.from_list(train_processed)
val_dataset = Dataset.from_list(val_processed)


tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])


model = AutoModelForSequenceClassification.from_pretrained(model_name_path, num_labels=len(label_map))
model.config.pad_token_id = tokenizer.pad_token_id


LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    task_type="SEQ_CLS",
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir=checkpoint_path,
    logging_dir=log_save_path,
    logging_steps=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=6,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    remove_unused_columns=False,
    fp16=False,
)


class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, threshold=0.01, metric_name="eval_accuracy"):
        self.patience = patience
        self.threshold = threshold
        self.metric_name = metric_name
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.previous_score = None

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):

        current_score = kwargs["metrics"][self.metric_name]
        
        if self.previous_score is None:

            self.best_score = current_score
            self.previous_score = current_score
        else:

            if self.previous_score != 0:
                improvement_rate = (current_score - self.previous_score) / self.previous_score
            else:
                improvement_rate = float('inf')


            self.previous_score = current_score


            if improvement_rate > self.threshold:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    control.should_training_stop = True


        if self.best_score == current_score:
            control.should_save = True

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.early_stop:
            print(f"Early stopping triggered after {state.global_step} steps")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[CustomEarlyStoppingCallback(patience=2, threshold=0.001, metric_name="eval_accuracy")]
)


trainer.train()
trainer.save_model(model_save_path)

eval_results = trainer.evaluate()
with open(os.path.join(result_save_path, "eval_results.txt"), "w") as f:
    f.write(f"Validation Loss: {eval_results['eval_loss']:.4f}\n")
    f.write(f"Accuracy: {eval_results['eval_accuracy']:.4f}\n")
    f.write(f"Precision: {eval_results['eval_precision']:.4f}\n")
    f.write(f"Recall: {eval_results['eval_recall']:.4f}\n")
    f.write(f"F1-score: {eval_results['eval_f1']:.4f}\n")

print(f"模型 {model_name} 微调完成，结果已保存至 {model_folder}")
# import transformers
# import datasets
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed)
import nltk
import numpy as np

from conf import Config
# ==============================================
# FINE-TUNING MODEL ON SUMMARIZATION TASK
# ==============================================
set_seed(Config.TRAIN["seed"])
model_checkpoint = Config.TRAIN["model_checkpoint"]

# load dataset
raw_datasets = load_dataset(Config.TRAIN["dataset_name"], Config.TRAIN["dataset_config_name"])
metric = load_metric("rouge")

# preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""
    
max_input_length = Config.TRAIN["max_source_length"]

max_target_length = Config.TRAIN["max_target_length"]

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples[Config.TRAIN["text_column"]]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[Config.TRAIN["summary_column"]],
                           max_length=max_target_length,
                           truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = Config.TRAIN["batch_size"]
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-xsum",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload
metrics = train_result.metrics

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


logger.info("*** Evaluate ***")
metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
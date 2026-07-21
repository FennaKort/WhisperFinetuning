https://medium.com/@chris.xg.wang/a-guide-to-fine-tune-whisper-model-with-hyper-parameter-tuning-c13645ba2dba includes padding audio directions; where to find thing I read about a better way to split and pad audio for finetuning?

https://github.com/Vaibhavs10/fast-whisper-finetuning/blob/main/README.md

## method from https://learnopencv.com/fine-tuning-whisper-on-custom-dataset/
additionally need to install:
`py -m pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio`


## [Load a Pre-Trained Checkpoint](https://huggingface.co/blog/fine-tune-whisper#load-a-pre-trained-checkpoint)

```
# ERROR: for each, shows error Cannot assign to attribute "x" for class "GenerationConfig"
  Attribute "x" is unknown
model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
``` 

## [Define a Data Collator](https://huggingface.co/blog/fine-tune-whisper#define-a-data-collator) 

## [Evaluation Metrics](https://huggingface.co/blog/fine-tune-whisper#evaluation-metrics) 
```
wer = 100*metric.compute(predictions=pred_str, references=label_str) # note that line as written by Gandi is wer = 100 * metric.compute(predictions=pred_str, references=label_str) but this gives a warning of potential problem that the operator * isn't supported between these data types
```

## [Define the Training Arguments](https://huggingface.co/blog/fine-tune-whisper#define-the-training-arguments) 
	dataset_sample_test = dataset # TODO 2026/07/20 only attempting to use this for the purposes of testing that the trainer can instantiate correctly
	---
	training_args = Seq2SeqTrainingArguments(
			output_dir="./fine-tuned-model",  # change to a repo name of your choice
			per_device_train_batch_size=16,
			gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
			learning_rate=1e-5,
			warmup_steps=500,
			max_steps=5000,
			gradient_checkpointing=True,
			fp16=True,
			eval_strategy="no", # Gandi guide says to set this to "steps", setting to 'no' while not passing an `eval_dataset` as per: ```ValueError: You have set `args.eval_strategy` to IntervalStrategy.STEPS but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. ```
			save_strategy='best', # as per ```ValueError: --load_best_model_at_end requires the save and eval strategy to match, except when --save_strategy="best", but found 	- Evaluation strategy: IntervalStrategy.NO 	- Save strategy: SaveStrategy.STEPS```
			per_device_eval_batch_size=8,
			predict_with_generate=True,
			generation_max_length=225,
			save_steps=1000,
			eval_steps=1000,
			logging_steps=25,
			report_to=["tensorboard"],
			load_best_model_at_end=True,
			metric_for_best_model="wer",
			greater_is_better=False,
			push_to_hub=False,
			)

	trainer = Seq2SeqTrainer(
		args=training_args,
		model=model,
		data_collator=data_collator,
		train_dataset=dataset, #training dataset
		#eval_dataset=dataset_sample_test, #testing dataset
		compute_metrics=finetuner.compute_metrics
	)

## [Training](https://huggingface.co/blog/fine-tune-whisper#training)
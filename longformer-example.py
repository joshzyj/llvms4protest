from transformers import LongformerTokenizer, LongformerForSequenceClassification, LongformerConfig, Trainer, TrainingArguments,LongformerTokenizerFast,LongformerModel
from datasets import Dataset
import pandas as pd
import os
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import collections 
from collections.abc import Iterable,Container

collections.Container = collections.abc.Container
os.environ["WANDB_DISABLED"] = "false"


# load tokenizer and define length of the text sequence

model_name_or_path = 'allenai/longformer-base-4096'
tokenizer = LongformerTokenizerFast.from_pretrained(model_name_or_path, max_length =4096)

# load fine-tuned model
model = LongformerForSequenceClassification.from_pretrained("./longformer-models/checkpoint-720")

model.config


# Here are some examples of using fine-tuned model to infer protest

inputs = tokenizer("After a heated week of police violence, protests erupted in several US cities on Friday, at times turning tense.In the wake of the killings of Daunte Wright, a 20-year-old shot by police after being pulled over, and unarmed 13-year-old Adam Toledo, thousands took to the streets to demonstrate, sometimes into the night.Police officers make an arrest during a rally calling for justice over the death of George Floyd, in Brooklyn, New York, on 1 June 2020 Most charges against George Floyd protesters dropped, analysis shows In Chicago, where Adam was killed, thousands marched in Logan Square after the video of the 13-year-old being shot with his arms raised was released this week. The protesters planned to march to Mayor Lori Lightfoot’s home, some of them calling for her resignation.The event was largely peaceful, though some police and protesters scuffled as the night drew to a close.In the Minneapolis suburb of Brooklyn Center, protesters staged the sixth straight night of demonstrations outside the police headquarters. Smaller demonstrations also occurred in downtown Minneapolis.Authorities had initially declined to declare an evening curfew, after quieter protests on the previous two nights. But after clashes between protesters and police in Brooklyn Center, during which authorities claimed a fence around the heavily fortified police headquarters was breached, an unlawful assembly was declared and 100 arrests were made.Earlier in the evening, a US district judge ruled that Minnesota state patrol could not arrest, threaten or target journalists after an ACLU complaint that law enforcement was unfairly cracking down on working reporters.But on Friday night a number of reporters documented being detained by police and released only after being photographed by officers with their press identification badges.Meanwhile, a protest that began peacefully in California ended with multiple fires set, several cars damaged and numerous windows shattered.The protest against police brutality in Oakland began calmly on Friday night, news outlets reported. A subsequent march drew about 300 people.", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]


inputs = tokenizer("A Texas substitute teacher was fired after she allegedly encouraged in-class fights at the Mesquite middle school where she taught, school officials said Friday.The educator, a substitute employed since early March, was terminated Thursday after the district discovered fights took place in her class at Kimbrough Middle School on Wednesday, Mesquite Independent School District said in a statement.Our investigation revealed that this substitute teacher encouraged students to fight each other during class, outlined rules for the students to follow and even instructed a student to monitor the classroom door while the fights took place, it said.The district characterized the teacher's actions as appalling and intolerable. The educator has not been named. The Mesquite Education Association did not immediately respond to a request for comment.The district said it has referred the matter to the Mesquite Police Department, where a spokesperson said an investigation was underway.", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]


inputs = tokenizer("This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.This is not a protest.", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]


inputs = tokenizer("Finance is a field that deals with the study of money and investments. It includes the dynamics of assets and liabilities over time under conditions of different degrees of uncertainty and risk.[20] In the context of business and management, finance deals with the problems of ensuring that the firm can safely and profitably carry out its operational and financial objectives; i.e. that it: (1) has sufficient cash flow for ongoing and upcoming operational expenses, and (2) can service both maturing short-term debt repayments, and scheduled long-term debt payments. Finance also deals with the long term objective of maximizing the value of the business, while also balancing risk and profitability; this includes the interrelated questions of (1) capital investment, which businesses and projects to invest in; (2) capital structure, deciding on the mix of funding to be used; and (3) dividend policy, what to do with excess capital.", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

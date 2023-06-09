from transformers.models.auto import AutoConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
from mlflow.models.signature import infer_signature
import os, pathlib
from mlflow.pyfunc import PythonModel, PythonModelContext
from typing import Dict

# Set up MLflow experiment
mlflow.set_experiment("distilbert_text_classifier2")

from bert_wrapper import BertTextClassifier


model_uri = 'distilbert-base-uncased'
config = AutoConfig.from_pretrained(model_uri)
config.num_labels = 2

print('Architecture:', config.architectures)
print('Classes:', config.label2id.keys())


# Read sentences from .csv file, start with 1st row
df = pd.read_excel('/gcs/training-data-mlflow/ready/2502_sentences.xlsx', header=0, names=['label', 'sentence'])
# df = pd.read_excel('/Users/estebanfelipecaceresgelvez/Documents/tesis/datasets/2502_sentences.xlsx', header=0, names=['label', 'sentence'])


# df = df.dropna()

# Set your training data and labels
train_data = df["sentence"].tolist()  # Your training data
train_labels = df["label"].astype('int').tolist()  # Your training labels


# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the hyperparameters
epochs = 2
batch_size = 16
learning_rate = 2e-5

# Log hyperparameters
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("num_epochs", epochs)

# Initialize the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define the training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    dataloader = DataLoader(list(zip(train_data, train_labels)), batch_size=batch_size, shuffle=True)
    
    for inputs, labels in dataloader:
        inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(**inputs)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
    # Log metrics to MLflow
    mlflow.log_metric("loss", avg_loss, epoch + 1)

# Save the trained model and log its artifacts to MLflow
# mlflow.pytorch.save_model(model, "model")


print("Finished training. Saving model...")

# Log the model to MLflow
_ = model.eval()


# Getting structure of inputs and outputs
sample = pd.DataFrame({ 'text': ['The displacement was 11.8 ± 7.1 mm',
                                 'Then, the target was removed and a new target or the same target was present after 10 consecutive steps and remained for 10 steps']})

inputs = tokenizer(list(sample['text'].values), padding=True, return_tensors='pt')

for key in inputs.keys():
    inputs[key] = inputs[key].to(device)

predictions = model(**inputs)
probs = torch.nn.Softmax(dim=1)(predictions.logits)
probs = probs.detach().cpu().numpy()
classes = probs.argmax(axis=1)
confidences = probs.max(axis=1)
outputs = pd.DataFrame({ 'rating': [config.id2label[c] for c in classes], 'confidence': confidences })

# Saving the model
signature = infer_signature(sample, outputs)
# model_path = 'numeric_classifier'
model_path = '/gcs/models-mlflow'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Createing artifacts

artifacts = { pathlib.Path(file).stem: os.path.join(model_path, file) 
             for file in os.listdir(model_path) 
             if not os.path.basename(file).startswith('.') }

# Log the model to MLflow
# mlflow.set_experiment('bert-classification')


mlflow.pyfunc.log_model('classifier', 
                        python_model=BertTextClassifier(), 
                        artifacts=artifacts, 
                        signature=signature,
                        registered_model_name='bert-rating-classification')
from mlflow.pyfunc import PythonModel, PythonModelContext
from typing import Dict
import torch
import pandas as pd

class BertTextClassifier(PythonModel):
    def load_context(self, context: PythonModelContext):
        import os
        from transformers.models.auto import AutoConfig, AutoModelForSequenceClassification
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        
        config_file = os.path.dirname(context.artifacts["config"])
        self.config = AutoConfig.from_pretrained(config_file)
        self.tokenizer = AutoTokenizer.from_pretrained(config_file)
        self.model = AutoModelForSequenceClassification.from_pretrained(config_file, config=self.config)
        
        if torch.cuda.is_available():
            print('[INFO] Model is being sent to CUDA device as GPU is available')
            self.model = self.model.cuda()
        else:
            print('[INFO] Model will use CPU runtime')
        
        _ = self.model.eval()
        
    def _predict_batch(self, data):
        import torch
        import pandas as pd
        
        with torch.no_grad():
            inputs = self.tokenizer(list(data['text'].values), padding=True, return_tensors='pt', truncation=True)
        
            if self.model.device.index != None:
                torch.cuda.empty_cache()
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(self.model.device.index)

            predictions = self.model(**inputs)
            probs = torch.nn.Softmax(dim=1)(predictions.logits)
            probs = probs.detach().cpu().numpy()

            classes = probs.argmax(axis=1)
            confidences = probs.max(axis=1)

            return classes, confidences
        
    def predict(self, context: PythonModelContext, data: pd.DataFrame) -> pd.DataFrame:
        import math
        import numpy as np
        
        batch_size = 64
        sample_size = len(data)
        
        classes = np.zeros(sample_size)
        confidences = np.zeros(sample_size)

        for batch_idx in range(0, math.ceil(sample_size / batch_size)):
            bfrom = batch_idx * batch_size
            bto = bfrom + batch_size
            
            c, p = self._predict_batch(data.iloc[bfrom:bto])
            classes[bfrom:bto] = c
            confidences[bfrom:bto] = p
            
        return pd.DataFrame({'rating': [self.config.id2label[c] for c in classes], 
                             'confidence': confidences })  
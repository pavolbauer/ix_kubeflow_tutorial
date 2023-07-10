from kserve import Model, ModelServer
#from torchvision import models, transforms
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class Kserve_BERT_QA_Model(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.tokenizer = None

    def load(self):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]

        source_text = inputs[0]["text"]
        questions = inputs[0]["questions"]
        results = {}

        for question in questions:
            print("Processing question: " + question)
            inputs = self.tokenizer.encode_plus(question, source_text, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = self.model(**inputs)[0], self.model(**inputs)[1]
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])

            results[ question ] = answer

        return {"predictions": results}

if __name__ == "__main__":
    model = Kserve_BERT_QA_Model("huggingface")
    ModelServer().start([model])
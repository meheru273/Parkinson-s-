import torch
import torch.nn as nn 
from transformers import BertTokenizer, BertModel

class TextTokenizer(nn.Module):
    
    def __init__(self, model_name= 'bert-base-uncased',output_dim = 128,dropout = 0.1):
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        input_dim = self.bert.config.hidden_size

        self.face = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self,text_list,device):

        tokens = self.tokenizer(text_list, padding=True, truncation=True,max_length=512, return_tensors="pt")
        
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        output = outputs.pooler_output
        
        return self.face(output)

from torch import nn
import torch
from transformers import AutoModel

class CustomCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.dropout = 0.2

        embedding_model = AutoModel.from_pretrained('roberta-base')
        self.hidden_size = embedding_model.config.hidden_size
        self.embed = embedding_model.embeddings.word_embeddings

        # self.embed = nn.Embedding(50265, 512)

        self.conv = nn.Sequential(
            nn.Conv1d(self.hidden_size, 512, 1), nn.GELU(), nn.Dropout(self.dropout),
            nn.Conv1d(512, 2048, 3), nn.GELU(), nn.Dropout(self.dropout)
        )

        self.pooler = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.embed(input_ids).transpose(1, 2)
        x = self.conv(x)
        # x = torch.max(torch.stack([self.conv2(x[:,:,0:-2:3]), self.conv2(x[:,:,1:-1:3]), self.conv2(x[:,:,2::3])], dim=0), dim=0)[0]
        x = self.pooler(x).squeeze(-1)
        logits = self.classifier(x)

        loss = None
        if labels != None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}

from torch import nn
from transformers import AutoModel

class CustomCNN(nn.Module):
    def __init__(self, channels, classes=2, kernels=(8, 5), dropout=0.0, embedding_model='distilbert-base-uncased'):
        super().__init__()

        embedding_model = AutoModel.from_pretrained(embedding_model)
        hidden_size = embedding_model.config.hidden_size

        self.embed = embedding_model.embeddings.word_embeddings

        layers = []
        layers.append(nn.Conv1d(hidden_size, channels[0], kernels[0]))
        layers.append(nn.GELU())

        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv1d(in_c, out_c, kernels[1], padding=5))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            # layers.append(nn.AdaptiveMaxPool1d(1))

        self.net = nn.Sequential(*layers)

        self.pooler = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Linear(channels[-1], classes)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.embed(input_ids).transpose(1, 2)
        x = self.net(x)
        x = self.pooler(x).squeeze(-1)
        logits = self.classifier(x)

        loss = None
        if labels != None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}

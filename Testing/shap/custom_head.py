from torch import nn
from transformers import AutoModel

class CustomRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:,0,:]  # [CLS] token
        logits = self.regressor(hidden_state).squeeze(-1)  # shape: (batch,)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

class CustomConvolution(nn.Module):
    def __init__(self, model_name, num_classes=2, num_filters=100):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.convolution1 = nn.Conv1d(
            in_channels=self.bert.config.hidden_size,
            out_channels=num_filters,
            kernel_size=5,
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)
        self.convolution2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:,1:,:]  # omit [CLS] token - shape: (batch, seq_len, hidden_size)

        x = x.transpose(1, 2) # for convolution input - shape: (batch, hidden_size, seq_len)
        x = self.convolution1(x)  # shape: (batch, num_filters, new_seq_len)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.convolution2(x) # shape: (batch, num_classes, new_seq_len)

        logits = x.max(dim=2).values

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

class CustomCNN(nn.Module):
    def __init__(self, channels, kernels=(8, 6), classes=2, dropout=0.0, embedding_model='distilbert-base-uncased'):
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

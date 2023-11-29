class TransformerClassifier(nn.Module):

    def __init__(
        self,
        d_model=768,
        n_classes=3,
        nhead=4,
        dim_feedforward=256,
        num_layers=4,
        dropout=0.1,
        activation="relu",
        classifier_dropout=0.1,
    ):

        super().__init__()

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=5000,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier1 = nn.Linear(d_model, d_model)
        self.classifier2 = nn.Linear(d_model, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.d_model = d_model

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        
        return self.softmax(x)
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64, 10)

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.contiguous().view(batch_size, 16, 49)

        x = self.patch_embed(x)
        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.fc(x)
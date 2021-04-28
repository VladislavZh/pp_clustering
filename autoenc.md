### Autoencoder architecture
Default value are in parentheses.
#### Encoder
<ol>
<li>Embedding(vocabulary size = max(id), embed.dim = 128) for categorical variable</li>
<li>LSTM(input.dim = embed.dim + dim of continuous variable, hidden.dim = 50, n.layers=1, dropout=0.5)</li>
<li>Fully Connected Layer(hidden.dim = 50, encoder.dim = 10)</li>
</ol>

#### Decoder
<ol>
<li>Fully Connected Layer(encoder.dim = 10, hidden.dim = 50)</li>
<li>LSTM(hidden.dim=50,input.dim = embed.dim + dim of continuous variable, n.layers=1, dropout=0.5)</li>
<li>Fully Connected Layer (embed.dim=128, vocab.size = max(id))
<li>Concat output of torch.argmax(logits) and non-embedded output of LSTM
</ol>

import time
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

# use cpu or gpu based on your system
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

data_dir = "data.txt"
text = open(data_dir, "r").read()  # load all the data as simple string

# convert our text data into tokenized tensor
data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)

train_batch_size = 16  # training batch size
eval_batch_size = 8  # evaluation batch size
context_length = 512  # number of tokens processed in a single batch
train_split = 0.7  # percentage of data to use from total data for training


# used to define size of embeddings
d_model = 768
n_heads = 4  # number of self-attention heads. should be divisible with d_model
n_layers = 8  # number of gpt blocks/layers


# training setup
epochs = 2000
eval_steps = 500  # perform evaluation in every n steps
lr = 1e-3

# split data into trian and eval
n_data = len(data)
train_data = data[: int(n_data * train_split)]
eval_data = data[int(n_data * train_split) :]


class DataLoader:
    def __init__(self, tokens, batch_size, context_length) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length

        self.current_position = 0

    def get_batch(self) -> torch.tensor:
        b, c = self.batch_size, self.context_length

        start_pos = self.current_position
        end_pos = self.current_position + b * c + 1

        # if the batch exceeds total length, get the data till last token
        # and take remaining from starting token to avoid always excluding some data
        add_data = (
            -1
        )  # n, if length exceeds and we need `n` additional tokens from start
        if end_pos > len(self.tokens):
            add_data = end_pos - len(self.tokens)
            end_pos = len(self.tokens)

        d = self.tokens[start_pos:end_pos]
        if add_data != -1:
            d = torch.cat([d, self.tokens[:add_data]])

        x = (d[:-1]).view(b, c)  # inputs
        y = (d[1:]).view(b, c)  # targets

        self.current_position += b * c  # set the next position
        if self.current_position > len(self.tokens) - 1:
            self.current_position = 0
        return x, y


train_loader = DataLoader(train_data, train_batch_size, context_length)
eval_loader = DataLoader(eval_data, eval_batch_size, context_length)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert n_heads * self.head_dim == d_model

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape

        # Project the input embeddings into Q, K, and V
        Q = (
            self.query(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            self.key(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            self.value(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        # Apply mask to prevent attention to future tokens
        mask = (
            torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
            .bool()
            .to(inputs.device)
        )
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(self.dropout(attention_weights), V)

        # Concatenate heads and put them back to the original shape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)

        # Apply the final linear transformation
        out = self.fc_out(attention_output)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model) -> None:
        super().__init__()
        # Create a matrix of shape (context_length, d_model) to store the positional encodings
        pe = torch.zeros(context_length, d_model)

        # Create a vector with positions [0, 1, 2, ..., context_length-1] of shape (context_length, 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)

        # Create a vector with the divisor terms based on the dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, context_length, d_model)

        # Register pe as a buffer, so it is not considered a parameter but is part of the module's state
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add the positional encodings to the input embeddings
        return x + self.pe[:, : x.size(1), :]


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, logits):
        att_logits = self.att(logits)
        adn_logits = self.ln1(logits + att_logits)
        logits = self.dropout(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)
        return logits


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)  # word token embeddings
        self.wpe = PositionalEncoding(
            context_length, d_model
        )  # word position encodings
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.linear1 = nn.Linear(d_model, vocab_size)

        #parameter sharing
        self.wte.weight = self.linear1.weight

    def forward(self, inputs, targets=None):
        logits = self.wte(inputs)  # dim -> batch_size, sequence_length, d_model
        logits = self.wpe(logits)
        for block in self.blocks:
            logits = block(logits)
        logits = self.linear1(logits)
        loss = None
        if targets != None:
            batch_size, sequence_length, d_model = logits.shape
            # to calculate loss for all token embeddings in a batch
            # kind of a requirement for cross_entropy
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # this will store the model outputs along with the initial input sequence
        # make a copy so that it doesn't interfare with model
        output = inputs.clone()
        for _ in range(max_new_tokens):
            current_seq_length = inputs.size(1)
            # Truncate inputs if it exceeds context_length
            if current_seq_length > context_length:
                inputs = inputs[:, -context_length:]
            # we only pass targets on training to calculate loss
            logits, _ = self(inputs)
            # for all the batches, get the embeds for last predicted sequence
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            # get the probable token based on the input probs
            idx_next = torch.multinomial(probs, num_samples=1)

            inputs = torch.cat([inputs, idx_next], dim=1)
            output = torch.cat([output, idx_next], dim=1)
        return [tokenizer.decode(out.tolist()) for out in output]


m = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
m = torch.compile(m)


optim = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=3000, eta_min=lr*0.1)

for e in range(epochs):
    xb, yb = train_loader.get_batch()

    logits, loss = m(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1)

    optim.step()
    scheduler.step()

    if e % eval_steps == 0 or e == epochs-1:
        m.eval()
        with torch.no_grad():
            xvb, yvb = eval_loader.get_batch()
            _, e_loss = m(xvb, yvb)

        print(f"Epoch: {e}\ttrain_loss: {loss:.4f}\teval_loss: {e_loss:.4f}")

        m.train()  # Back to training mode

# saying to torch that do not store gradients for whatever we do below
with torch.no_grad():
    input = torch.tensor(
        tokenizer.encode("Love "), dtype=torch.long, device=device
    ).unsqueeze(0)
    print(m.generate(input, max_new_tokens=500)[0])

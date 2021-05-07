# Machine Learning deploy sentiment analysis model

Second project from Udacity's Machine Learning Engineer Nanodegree building and deployment a Sentiment Analysis Model

Project sections:

- Problem understanding
- Project structure
- Running experiments
- Results report

## Problem understanding

Build a recurrent neural network for the purpose of determining the sentiment of a movie review using the IMDB data set using Amazon's SageMaker service. In addition, you will deploy your model and construct a simple web app which will interact with the deployed model.


## Project structure

The project structure is based on the Udacity's project template:

```
+ serve: aima code library    + model.py            LSTMClassifier class implementation for producition
                              + predict.py          Service implementation using LSTMClassifier for the inference process
                              + utils.py            Utils function for text preparation (tokenizer and padding)
                              + requirements.txt    Python libraries required 

+ train                       + model.py            LSTMClassifier class implementation for training
                              + predict.py          Service implementation using LSTMClassifier for the inference process
                              + requirements.txt    Python libraries required 

+ website                     + index.html          Web form to dall the endpoint service using a POST call
+ SageMaker Project.ipynb                           Notebook with implementation for the LSTM sentiment analysis model

```

## Implementation

LSTM Class implementation

  ```Python
  import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        # Dropout Layer
        self.dropout = nn.Dropout(0.3)
        
        # Dense Layer
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        
        #Sigmoid Layer
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())
  ```

Train function implementation

  ```
  def train(model, train_loader, epochs, optimizer, loss_fn, device):
      """
      This is the training method that is called by the PyTorch training script. The parameters
      passed are as follows:
      model        - The PyTorch model that we wish to train.
      train_loader - The PyTorch DataLoader that should be used during training.
      epochs       - The total number of epochs to train for.
      optimizer    - The optimizer to use during training.
      loss_fn      - The loss function used for training.
      device       - Where the model and data should be loaded (gpu or cpu).
      """

      # TODO: Paste the train() method developed in the notebook here.

      # Clipping parameter
      clip=5 

      for epoch in range(1, epochs + 1):
          model.train()
          total_loss = 0
          for batch in train_loader:         
              batch_X, batch_y = batch

              batch_X = batch_X.to(device)
              batch_y = batch_y.to(device)

              # TODO: Complete this train method to train the model provided.

              # Zero accumulated gradients
              optimizer.zero_grad()

              #  Execute a forward pass
              output = model.forward(batch_X)

              # calculate batch loss using the loss functions provided
              loss = loss_fn(output, batch_y)

              # Execure the backpropagation
              loss.backward()

              # Apply gradient clipping to avoid gradient exploding
              torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

              #Execute optimizer step
              optimizer.step()

              total_loss += loss.data.item()
          print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
  ```


Can run experiment with different parameters values:

  ```
  •	Time limit: 100 y 150 ms
  •	Opponent model:  GREEDY, MINIMAX, SELF, RANDOM
  •	Depth: 3, 5, 7 (Parameter DEFAULT_DEPTH inside my_custom_player.py)
  •	Matches: 100 (50 rounds)
  ```

To facilitate the execution was created a shell script named experiments.sh that executes these calls:
  ```
  python run_match.py -r 50 -o GREEDY -t 100
  python run_match.py -r 50 -o MINIMAX -t 100
  python run_match.py -r 50 -o SELF -t 100
  python run_match.py -r 50 -o RANDOM -t 100
  python run_match.py -r 50 -o GREEDY -t 150
  python run_match.py -r 50 -o MINIMAX -t 150
  python run_match.py -r 50 -o SELF -t 150
  python run_match.py -r 50 -o RANDOM -t 150
  ```

It's necessary assign the right permits to execute the script: 
  ```
  chmod u+x experiments.sh
  ```

## Results report


[Results report document](https://github.com/Fer-Bonilla/Udacity-Artificial-Intelligence-forward-planning-agent/blob/main/report.pdf)


## Author 
Fernando Bonilla [linkedin](https://www.linkedin.com/in/fer-bonilla/)

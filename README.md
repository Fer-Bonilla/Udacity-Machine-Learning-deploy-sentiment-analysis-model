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

To run the experiments, execute the `run_match.py` python script: 

  - Run the search experiment manually (you will be prompted to select problems & search algorithms)
  ```
  $ python run_match.py -r 50 -o GREEDY -t 150
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

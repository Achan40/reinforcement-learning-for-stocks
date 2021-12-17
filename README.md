# reinforcement-learning-for-stocks
Creating a reinforcement learning model to trade a single stock. 

### Dependencies
Python 3.7 To install the necessary libraries, run `pip install -r requirements.txt`

### Table of Contents
* `agent.py`: 
* `env.py`: 
* `model.py`: 
* `utils.py`: some utility functions
* `run.py`: 
* `requirements.txt`: all dependencies
* `data/`: data

### Commands
* Loading data from API
`python main.py --getdata <ticker> <timeframe> <version>`

`<ticker>` is the ticker symbol for a certain stock the only required arguement for this command, `<timeframe>` is a period of time. See [IEX Cloud API docs](https://iexcloud.io/docs/api/#historical-prices) for available arguements. `<version>` can be `sandbox` or `stable`. To run this command, you'll need an IEX Cloud account for the API keys. Then, create a `sandbox_secret.py` file and a `secret.py` file and add them to the projects root directory. Include a `SECRET_KEY` variable within them, which is the string representation of your sandbox API key and your API key for IEX Cloud. This command will create a directory `data/`, and save a `<ticker>.csv` file within it.

* Training Deep Q agent
`python main.py --mode train --dataset <dataset>`
Run model training. Additional arguements include: `--episode` (number of episodes to run), `--batch_size` (batch size for experience replay), `--initial_invest` (amount of initial investment), `--dataset` (name of the dataset to use). Note that a checkpoints for the model weights are created every 5 episodes.

* Testing Deep Q agent
`python main.py --dataset <dataset> --mode test --weights <trained_model>`
Test the model using the test split of a dataset. `<trained_model>` is the file location of the trained model (should be located in the weights directory which is created upon the first run of the script).

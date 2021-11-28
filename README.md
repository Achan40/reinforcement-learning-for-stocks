# reinforcement-learning-for-stocks
Creating a reinforcement learning model to trade a single stock. 

### Dependencies
Python 3.7 To install the necessary libraries, run `pip install -r requirements.txt`

### Table of Contents
* `agent.py`: 
* `envs.py`: 
* `model.py`: 
* `utils.py`: some utility functions
* `run.py`: 
* `requirements.txt`: all dependencies
* `data/`: data

### How to run
To gather historical data for a specific stock from IEX Cloud API, run: `python main.py --getdata <ticker> <timeframe> <version>`. `<ticker>` is the ticker symbol for a certain stock the only required arguement for this command, `<timeframe>` is a period of time. See [IEX Cloud API docs](https://iexcloud.io/docs/api/#historical-prices) for available arguements. `<version>` can be `sandbox` or `stable`. To run this command, you'll need an IEX Cloud account for the API keys. Then, create a `sandbox_secret.py` file and a `secret.py` file and add them to the projects root directory. Include a `SECRET_KEY` variable within them, which is the string representation of your sandbox API key and your API key for IEX Cloud.
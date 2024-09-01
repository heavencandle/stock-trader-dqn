# Stock Trading System 

## Overview

This repository contains a Python-based stock trading system designed for automating and simulating trading strategies. The system interfaces with external trading APIs to collect real-time market data, execute trades, and backtest strategies in a simulated environment. It's a versatile framework suitable for both live trading and historical data analysis, making it an essential tool for traders looking to automate their trading operations.

## Features

- **Automated Data Collection:** The system automatically collects real-time market data from supported APIs, enabling continuous analysis and decision-making.
  
- **Trading Strategies:** Implements a variety of customizable trading strategies. The strategies are executed by a trading agent that makes decisions based on current market conditions and predefined rules.

- **Simulation and Backtesting:** Offers a simulation environment to test trading strategies against historical data, allowing users to refine their approaches before deploying them in a live market.

- **API Integration:** Supports integration with popular trading APIs (e.g., Kiwoom) to facilitate real-time trading and data retrieval.

- **Data Processing and Visualization:** Includes tools for processing market data and visualizing both the data and the performance of trading strategies. This helps in understanding market trends and the impact of different strategies.

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/stock-trading-system.git
   cd stock-trading-system
   ```

2. **Install Dependencies**
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the System**
Start the trading system or a simulation:
  ```bash
  python main.py
  ```

## Usage
- Live Trading: Connect the system to a supported broker's API for real-time trading.
- Backtesting: Run the simulation mode using historical data to test and refine trading strategies before deploying them in a live environment.

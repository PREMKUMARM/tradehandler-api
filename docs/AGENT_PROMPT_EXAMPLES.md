# AI Agent Prompt Examples (Institutional VWAP Strategy)

The agent now follows a single, high-conviction **Institutional VWAP Strategy** for all trade analysis and decisions.

## Institutional VWAP Trading Opportunities

### Core VWAP Strategy Queries

1. **Daily Opportunity Analysis**
   ```
   what are the trades i would have been taken today in reliance based on vwap setup
   ```
   ```
   show me trading opportunities in Nifty today using the institutional vwap strategy
   ```
   ```
   what vwap trades could I have taken in TCS today
   ```
   ```
   find all institutional vwap signals in BankNifty today
   ```

2. **Multiple Instruments & Groups Analysis**
   ```
   find vwap trading opportunities in reliance, tcs, and infosys today
   ```
   ```
   what are the vwap trades in top 10 nifty50 stocks yesterday
   ```
   ```
   show me institutional vwap opportunities in nifty top 10 for the last 1 week
   ```
   ```
   analyze vwap trades in top 10 nifty50 stocks last 1 month
   ```

3. **Date Range Analysis**
   ```
   show me all vwap strategy trades in Reliance for the last 1 week
   ```
   ```
   what trades were possible in Nifty this month using institutional vwap
   ```
   ```
   analyze vwap opportunities in TCS for the last 5 days
   ```

4. **Specific Historical Date**
   ```
   what vwap trades would I have taken yesterday in Reliance
   ```
   ```
   show me vwap trading opportunities in Nifty on 2024-12-24
   ```

## Strategy Rules Applied Automatically

The agent automatically applies these institutional rules for every analysis:

1. **Trend Confirmation**: Price must be consistently above/below the daily VWAP.
2. **Pullback Zone**: Entry is only considered within **VWAP ± 0.2%**.
3. **Momentum Filter (RSI)**: 
   - **Buy**: RSI must be **below 40** (pullback) and turning up.
   - **Sell**: RSI must be **above 60** (pullback) and turning down.
4. **Candle Confirmation**: Looks for **Hammer**, **Engulfing**, or **Rejection** patterns at the VWAP line.
5. **Risk Management**:
   - **Stop Loss**: Below VWAP or recent swing low.
   - **Target**: Fixed **1:3 Risk:Reward Ratio**.
   - **Capital**: Uses your configured capital (default: ₹2,00,000) and risk (1% per trade).

## Market Analysis Queries

1. **Current VWAP Status**
   ```
   what is the current price of Reliance relative to its VWAP
   ```
   ```
   is Nifty trading above or below its VWAP right now
   ```

2. **Trend & Indicators**
   ```
   analyze the trend and vwap for Reliance
   ```
   ```
   what is the current RSI and VWAP for TCS
   ```

## Portfolio & Position Queries

1. **Current Status**
   ```
   show my current positions
   ```
   ```
   what is my account balance
   ```
   ```
   get my portfolio summary
   ```

2. **Risk Assessment**
   ```
   calculate risk for buying 100 shares of Reliance at 2450
   ```
   ```
   what is the position size for a trade in TCS with ₹500 risk
   ```

## Trading Commands

1. **Order Execution**
   ```
   buy 10 lots of NIFTY24JAN24500CE at market price
   ```
   ```
   sell 100 shares of RELIANCE at limit price 2450
   ```

2. **Exiting**
   ```
   exit all my Reliance positions
   ```
   ```
   close my losing positions
   ```

## Tips for Best Results

1. **Focus on "Trades" or "Opportunities"**: The agent will automatically use the VWAP framework for these keywords.
2. **Specify the Instrument and Date**: To get the most accurate back-analysis.
3. **Configuration**: The agent uses your **Trading Capital (₹2,00,000)** and **Risk (1%)** settings automatically to calculate Quantity.

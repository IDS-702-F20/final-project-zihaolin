# Final Project - 

This project aims to analyze the time series characteristics of every day profit and forecast the real volatility of S&P 500 index. Using ARMA model, I found that AR(2) model could fit the profit of stock price best however the diagnostics shows that the AR(2) is not good enough. Then I use ARIMA(2, 0, 0)-GARCH(1,1) model to fit the error term of profit and then use this model to predict the volatility of profit. The final result shows that the correlation between predicted value of volatility and realized volatility is 79.6% which is a good result.

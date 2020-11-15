
rm(list=ls())

library(lubridate)
library(tseries)
library(moments)
library(MASS)
library(forecast)
library(rugarch)

################################################################
#################### aim of this project #######################
################################################################

# This project aims to analyze the time series characteristics of every day profit 
# and forecast the real volatility of SP 500 index. 
# The S&P 500 is a stock market index that measures the stock performance of 500 
# large companies listed on stock exchanges in the United States. 

# We will use ARMA model to analyze the time series characteristics of profits 
# and use GARCH model to predict the volatility of profits.

# The realized volatility is invisible. Therefore, we should first estimate the real volatility.
# Some researcher [1] created a HEAVY model, Factor High-Frequency-Based Volatility, which is applied 
# to estimate volatility. The dataset I used has included the estimated valie (SPX2.rvol).

# [1] Andersen, Bollerslev Diebold (2008), Barndorff-Nielsen and Shephard (2007) and 
# Shephard and Sheppard (2009).

# Before we start the analysis, we need to make an assumption: the HEAVY estimates is nonbiased and
# efficiency. In the following project, I would use HEAVY estimates as "observed real volatility"
# to check whether my forecasting algorithm is good enough.

# The dataset comes from SPX every day data, including open price, close price, etc. The dataset has realized
# the Shephard and Sheppard model to estimate SPX real volatility. SPX2.rv is the real variance of 
# estimation, SPX2.r is the profit of each day and SPX2.rvol is the real volatility.



################################################################
###################### import the data #########################
################################################################

SPXdata<- read.csv('/Users/linqianzhi/Desktop/IDS\ 702/Project/final\ project\ drafts/SPX_rvol.csv')
rownames(SPXdata)<- ymd(SPXdata$DATE)  # make date to be the row nae 
SPXdata$SPX2.rvol<- sqrt(SPXdata$SPX2.rv)
SPXdata_1 <- SPXdata[,c('SPX2.r', 'SPX2.rv', 'SPX2.rvol', 'SPX2.openprice', 'SPX2.closeprice', 'DATE')]

################################################################
####### Step1: analyze the time series characteristics #########
################################################################

# Part0: volatility

## preliminary analysis for SPX2.rv 
# acf(SPXdata$SPX2.rvol)
# pacf(SPXdata$SPX2.rvol)


# adf.test(SPXdata$SPX2.rvol)
# kpss.test(SPXdata$SPX2.rvol, null = 'Level')
# ADF and KPSS test get both rejected. It is a flag of LRD.
# plot(SPXdata$DATE, SPXdata$SPX2.rvol, type = 'n')
# lines( SPXdata$DATE, SPXdata$SPX2.rvol)

# Analyze the every day profit time series characteristics

## Fact:
## 1. SPX2.r shows week auto corr and SPX2.rv shows strong LRD (long range dependency)
## 2. SPX2.r accepts the hypo of stationary
## 3. SPX2.r rejects the hypo of normality, but obey t-distribution

# 1. See the correlation characteristics
par(mfcol=c(2,1))
acf(SPXdata$SPX2.r)  # estimate MA part, q, MA(0)
pacf(SPXdata$SPX2.r)  # estimate RA part, p, RA(2)

jpeg('profit_acf_pacf.jpeg')
par(mfcol=c(2,1))
acf(SPXdata$SPX2.r)  # estimate MA part, q, MA(0)
pacf(SPXdata$SPX2.r)  # estimate RA part, p, RA(2)
dev.off()

Box.test(SPXdata$SPX2.r, lag =2 , type= 'Ljung-Box')  
    # Box-Ljung test: exists self-correlation, p-value < 0.05, reject null hypo 

# 2. Check the assumption of stationary
adf.test(SPXdata$SPX2.r)  # Dickey-Fuller Test: stationary, p-value 0,01, accept alternative hypo
kpss.test(SPXdata$SPX2.r, null = 'Level')  # KPSS Test: stationary, p-value = 0.1, accept null hypo

# 3. Check the assumption of normality
shapiro.test(SPXdata$SPX2.r)  # p-value < 0.05, reject null hypo, not normality
plot(density(SPXdata$SPX2.r))
kurtosis(SPXdata$SPX2.r)  # 10.56322, leptokurtosis and heavy tails

## model SPX2.r by t dist
# Fit the every day profit by t-distribution. The black line is the density of SPX2.r
# The green line is density of t-distribution after zooming.
par(mfcol=c(1,1))
hist(SPXdata$SPX2.r, pch=20, breaks=25, prob=TRUE, main="")

t.pars<-fitdistr(SPXdata$SPX2.r, densfun = 't', start= list(m=0,s= 0.01 ,df= 1))
plot(density(SPXdata$SPX2.r), xlim= c(-.1,.1), ylim=c(-1, 55) )
par(new=TRUE)
curve( dt( (x- t.pars$estimate[1])/t.pars$estimate[2], 
           df= t.pars$estimate[3])/ t.pars$estimate[2],
       from= -.1,
       to= .1, xlim= c(-.1,.1), 
       ylim=c (-1, 55),
       col= 'green',
       ylab = '')

################################################################
######## Step2: Fit the model of profit and diagnostics ########
################################################################


auto.arima(SPXdata$SPX2.r)
## auto.arima indicates AR(2) model for SPX2.r
# Next, we do the model diagnostics for the ARMA model of profit.
Model1.r <- arima(SPXdata$SPX2.r, order = c(2,0,0))
Model1.r
plot(SPXdata$SPX2.r)
test_data <- SPXdata[0:300,]
tsprofut <- ts(test_data$SPX2.r)
ts.plot(tsprofut,col="red3")

# 1. 2. 3. residual analysis, acf of residuals, The independence of residuals: The Ljung-Box Test
tsdiag(Model1.r, gof=15, omit.initial=F)
# The command tsdiag displays three of our diagnostic tools in one display -- a sequence plot
# of the standardized residuals, the sample ACF of the residuals, and p-value for the Ljung-Box
# test statistic for a whole range of values of lags from  1 to 15.

# 1. residual analysis
# In plot1, there are many residuals (more than 10) in the series with magnitudes larger than 3
# which is very unusual in a standard normal distribution.

# 2. acf and pacf of residuals
# There is no evidence of autocorrelation in the residuals of this model.

# 3. The independence of residuals: The Ljung-Box Test
# We would reject the ARMA(p, q) model if the observed value of Q* exceeded an appropriate
# critical value in a chi-square distribution with K-p-q degrees of freedom.
# We can see that when lag is more than 4, the p-value of Ljung-Box statistic is near 0,
# which means that we should reject the null hypo, and there exists self-correlation.

# 4. qq-norm
qqnorm(residuals(Model1.r))
qqline(residuals(Model1.r))
shapiro.test(residuals(Model1.r))  # p-value < 0.05, reject null hypo, not normality
# QQ norm plot shows that the residuals does not obey the normal distribution.

# Above all, the model is not good enough. However, our final goal is to predict the
# volatility of profit and the model is given by auro.arima, therefore here we just 
# remain the AR(2) model of the profit and use it to for the model predicting the volatality.



################################################################
###### Step3: Fit the model for predicting the volatility ######
################################################################


# Our goal is to estimate the variance of profit, however simple ARMA model could not help us.
# Therefore, we introduce the GARCH model, generalized autoregressive conditional heteroskedasticity model
# which is based on ARCH model, autoregressive conditional heteroscedasticity model.
# ARCH model is a statistical model for time series data that describes the variance of the 
# current error term or innovation as a function of the actual sizes of the previous time periods'
# error terms[2]; often the variance is related to the squares of the previous innovations. 
# The ARCH model is appropriate when the error variance in a time series follows an AR model.
# If an ARMA model is assumed for the error variance, the model is a GARCH model[3].

# ARCH models are commonly employed in modeling financial time series that exhibit time-varying 
# volatility and volatility clustering, i.e. periods of swings interspersed with periods of relative calm.
# Followed by 'GARCH model.md' file.

# [2] Engle, Robert F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation". 
# Econometrica. 50 (4): 987–1007. doi:10.2307/1912773. JSTOR 1912773.

## So apply standard ARMA(2,0)-GARCH(1,1) model for SPX2.r

##################
## Part1: fit the model
# First, I used rugarch package to fit the GARCH model, ugarchspec function could do that. 
# Our goal is to predict the volatility, therefore, I used 4189 observations to do backtest 
# (from Jan. 3rd 2000 to Oct. 06th 2016). I used the first 1000 observations to train the model
# and each time rolled to predict the next value and refitted the model every five observations.



sgarch<- ugarchspec(mean.model = list( armaOrder= c(2,0),
                                       include.mean= TRUE),
                    distribution.model = 'std')
sgarch_fitted<- ugarchfit(sgarch, data =  SPXdata$SPX2.r)

whole_len<- length( SPXdata$SPX2.r)
burning_len<- 1000
forecast_len<- whole_len- burning_len
ret<- data.frame(SPX2.r= SPXdata$SPX2.r)
rownames(ret)<- rownames(SPXdata)
# rolling estimation and forecasting
sgarch_roll<- ugarchroll(spec = sgarch,
                         data= ret,
                         n.ahead = 1,
                         forecast.length =forecast_len,
                         refit.every = 5)

sgarch
##################
## Part2: plot the result
## plot the predicted vol and realized vol(5-min estimator by OxManLab)
# The following plot shows the realized volatility and predicted volatility.
library(ggplot2)
library(reshape2)
x<- tail( ymd( SPXdata$DATE), forecast_len)
realized_vol<- sqrt(tail( SPXdata$SPX2.rv, forecast_len))
sgarch.predicted_vol<- sgarch_roll@forecast$density[,'Sigma']
tmp_df<- data.frame(x, realized_vol, sgarch.predicted_vol)
sgarch.g<- ggplot(melt(tmp_df, id.var= 'x'), aes(x=x, y= value))+
  geom_line(aes(colour= variable, group= variable))+
  scale_color_manual(values = c('grey', 'red'))+
  ylab('daily volatility')+
  xlab('date index')+
  theme(legend.title= element_blank())+
  ggtitle('ARMA(2,0)-GARCH(1,1) vol prediction')

# plot the realized volatility and sgarch.predicted volatility
jpeg('ARMAGARCH.jpeg')
sgarch.g
dev.off()

#################
## Part3: compute the squared error

sgarch.MSE<- mean(( realized_vol- sgarch.predicted_vol)^2)
summary( (realized_vol-sgarch.predicted_vol)^2)
sgarch.MSE  # 1.767135e-05
cor( realized_vol, sgarch.predicted_vol)  # 0.7960698
summary( lm( realized_vol~ sgarch.predicted_vol))

## save sgarch model
sgarch_model<- list()
sgarch_model$spec<- sgarch
sgarch_model$roll<- sgarch_roll
sgarch_model$plot<- sgarch.g
sgarch_model$MSE<- sgarch.MSE
sgarch_model$roll.pred<- tmp_df
save(sgarch_model, file = 'sgarch_model')

sgarch_model$spec
sgarch_model$roll

##################
## Part4: model diagnostics
# 1. Normality of residuals
plot(sgarch_fitted, which=8)
plot(sgarch_fitted, which=9)
# good enough

# 2. independence of residuals
acf(residuals(sgarch_fitted))
plot(sgarch_fitted, which=10)
plot(sgarch_fitted, which=11)
# good enough

# 3. significance of coefficients
sgarch_fitted
# p-value smaller than 0.05, good enough

# 4. residuals
plot(sgarch_fitted, which=3)
plot(residuals(sgarch_fitted))
# good enough



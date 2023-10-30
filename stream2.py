import streamlit as st
import io 
import pandas as pd
import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import json
import yfinance as yf
import pandas_datareader as pdr
import random
import scipy as sc
import plotly.graph_objects as go
from tqdm import tqdm
import fscraper as fs
import riskfolio as rp
from reportlab.pdfgen import canvas 

# Define the function to get data
def getData(stocks, start, end):
    data = {}
    errors = []
    for stock in stocks:
        yfs = fs.YahooFinanceScraper(stock)
        try:
            df = yfs.get_stock_price2(start, end)
        except:
            errors.append(stock)
            continue
        data[stock] = list(df['close'].values)
    max_length = max(len(lst) for lst in data.values())
    for key in data.keys():
        data[key] = [0 for _ in range((max_length - len(data[key])))] + data[key]
    if len(errors) > 0:
        st.write(f'{errors}: No Data Found')
    returns = pd.DataFrame(data).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    meanReturns = returns.replace([np.inf, -np.inf], np.nan).fillna(0).mean()
    covMatrix = returns.cov()
    corrMatrix = returns.corr()
    stdReturns = returns.replace([np.inf, -np.inf], np.nan).fillna(0).std()
    return returns, meanReturns, stdReturns, covMatrix, corrMatrix

def sum_to_one(n):
    values = [0.0, 1.0] + [random.random() for _ in range(0, n - 1)]
    values.sort()
    return np.array([values[i+1] - values[i] for i in range(n)])

def simulatePortfolioPerformance(meanReturns, covMatrix, n=25, iters=10000):
    returnsList = []
    stdList = []
    weightsList = []

    for i in range(iters):
        weights = sum_to_one(n)
        returns, std = portfolioPerformance(weights, meanReturns, covMatrix)
        returnsList.append(returns)
        stdList.append(std)
        weightsList.append(weights)
        
    return returnsList, stdList, weightsList

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(252)
    return returns, std

def negativeSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate=0.05):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -((pReturns - riskFreeRate)/pStd)

def maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=0.05, constraintSet=(0,1)):
    """maximize sharpe ratio by minimizing the negative sharpe ratio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(negativeSharpeRatio, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def portfolioReturn(weights, meanReturns, covMatrix):
        return portfolioPerformance(weights, meanReturns, covMatrix)[0]

def minVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    """Portfolio Optimization by Finding Weights That Corresponding to Minimum Portfolio Variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def calculatedResults0(meanReturns, covMatrix, riskFreeRate=0.05, constraintSet=(0,1)):
    """Read Relavent Financial Information and Output Portfolios Optimized for Max. Sharpe Ratio, Min. Variance & the Efficient Frontier
    method = 'maxsr', 'minvar'
    """
    # Max Sharpe Ratio 
    maxSrPortfolio = maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=riskFreeRate)
    maxSrPortReturns, maxSrPortStd = portfolioPerformance(maxSrPortfolio.x, meanReturns, covMatrix)
    maxSrPortReturns, maxSrPortStd = round(maxSrPortReturns*100,2), round(maxSrPortStd*100,2)
    maxSrAllocation = pd.DataFrame(maxSrPortfolio.x, index=meanReturns.index, columns=['Allocation(%)'])
    maxSrAllocation['Allocation(%)'] = [(round(i, 4) )*100 for i in maxSrAllocation['Allocation(%)']]

    # Min Variance
    minVarPortfolio = minVariance(meanReturns, covMatrix)
    minVarPortReturns, minVarPortStd = portfolioPerformance(minVarPortfolio.x, meanReturns, covMatrix)
    minVarPortReturns, minVarPortStd = round(minVarPortReturns*100,2), round(minVarPortStd*100,2)
    minVarAllocation = pd.DataFrame(minVarPortfolio.x, index=meanReturns.index, columns=['Allocation(%)'])
    minVarAllocation['Allocation(%)'] = [(round(i, 4) )*100 for i in minVarAllocation['Allocation(%)']]
    return maxSrPortReturns, maxSrPortStd, maxSrAllocation, minVarPortReturns, minVarPortStd, minVarAllocation

def calculatedResultsChoose(meanReturns, covMatrix, riskFreeRate=0.05, method = 'maxsr', constraintSet=(0,1)):
    """Read Relavent Financial Information and Output Portfolios Optimized for Max. Sharpe Ratio, Min. Variance & the Efficient Frontier
    method = 'maxsr', 'minvar'
    """
    if method == 'maxsr':
        # Max Sharpe Ratio 
        maxSrPortfolio = maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=riskFreeRate)
        maxSrPortReturns, maxSrPortStd = portfolioPerformance(maxSrPortfolio.x, meanReturns, covMatrix)
        maxSrPortReturns, maxSrPortStd = round(maxSrPortReturns*100,2), round(maxSrPortStd*100,2)
        maxSrAllocation = pd.DataFrame(maxSrPortfolio.x, index=meanReturns.index, columns=['Allocation(%)'])
        maxSrAllocation['Allocation(%)'] = [(round(i, 4) )*100 for i in maxSrAllocation['Allocation(%)']]
        return maxSrPortReturns, maxSrPortStd, maxSrAllocation
    elif method == 'minvar':
        # Min Variance
        minVarPortfolio = minVariance(meanReturns, covMatrix)
        minVarPortReturns, minVarPortStd = portfolioPerformance(minVarPortfolio.x, meanReturns, covMatrix)
        minVarPortReturns, minVarPortStd = round(minVarPortReturns*100,2), round(minVarPortStd*100,2)
        minVarAllocation = pd.DataFrame(minVarPortfolio.x, index=meanReturns.index, columns=['Allocation(%)'])
        minVarAllocation['Allocation(%)'] = [(round(i, 4) )*100 for i in minVarAllocation['Allocation(%)']]
        return minVarPortReturns, minVarPortStd, minVarAllocation

def efficientOpt(meanReturns, covMatrix, returnTarget=0.1, constraintSet=(0,1)):
    """For each Target Return, Optimize Portfolio for Min. Variance"""
    
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = sc.optimize.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return effOpt

def calculatedResults1(meanReturns, covMatrix, riskFreeRate=0.05, constraintSet=(0,1)):
    """Read Relavent Financial Information and Output Portfolios Optimized for Max. Sharpe Ratio, Min. Variance & the Efficient Frontier
    method = 'maxsr', 'minvar'
    """
    # Max Sharpe Ratio 
    maxSrPortfolio = maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=riskFreeRate)
    maxSrPortReturns, maxSrPortStd = portfolioPerformance(maxSrPortfolio.x, meanReturns, covMatrix)
    maxSrPortReturns, maxSrPortStd = round(maxSrPortReturns*100,2), round(maxSrPortStd*100,2)
    maxSrAllocation = pd.DataFrame(maxSrPortfolio.x, index=meanReturns.index, columns=['Allocation(%)'])
    maxSrAllocation['Allocation(%)'] = [(round(i, 4) )*100 for i in maxSrAllocation['Allocation(%)']]

    # Min Variance
    minVarPortfolio = minVariance(meanReturns, covMatrix)
    minVarPortReturns, minVarPortStd = portfolioPerformance(minVarPortfolio.x, meanReturns, covMatrix)
    minVarPortReturns, minVarPortStd = round(minVarPortReturns*100,2), round(minVarPortStd*100,2)
    minVarAllocation = pd.DataFrame(minVarPortfolio.x, index=meanReturns.index, columns=['Allocation(%)'])
    minVarAllocation['Allocation(%)'] = [(round(i, 4) )*100 for i in minVarAllocation['Allocation(%)']]
    

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVarPortReturns, maxSrPortReturns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target).fun)
        
    return maxSrPortReturns, maxSrPortStd, maxSrAllocation, minVarPortReturns, minVarPortStd, minVarAllocation

def calculatedResults(meanReturns, covMatrix, riskFreeRate=0.05, sim=25, constraintSet=(0,1)):
    """Read Relavent Financial Information and Output Portfolios Optimized for Max. Sharpe Ratio, Min. Variance & the Efficient Frontier
    method = 'maxsr', 'minvar'
    """
    # Max Sharpe Ratio 
    maxSrPortfolio = maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=riskFreeRate)
    maxSrPortReturns, maxSrPortStd = portfolioPerformance(maxSrPortfolio.x, meanReturns, covMatrix)
    maxSrPortReturns, maxSrPortStd = round(maxSrPortReturns*100,2), round(maxSrPortStd*100,2)
    maxSrAllocation = pd.DataFrame(maxSrPortfolio.x, index=meanReturns.index, columns=['Allocation(%)'])
    maxSrAllocation['Allocation(%)'] = [(round(i, 4) )*100 for i in maxSrAllocation['Allocation(%)']]

    # Min Variance
    minVarPortfolio = minVariance(meanReturns, covMatrix)
    minVarPortReturns, minVarPortStd = portfolioPerformance(minVarPortfolio.x, meanReturns, covMatrix)
    minVarPortReturns, minVarPortStd = round(minVarPortReturns*100,2), round(minVarPortStd*100,2)
    minVarAllocation = pd.DataFrame(minVarPortfolio.x, index=meanReturns.index, columns=['Allocation(%)'])
    minVarAllocation['Allocation(%)'] = [(round(i, 4) )*100 for i in minVarAllocation['Allocation(%)']]
    

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(num = sim, start = minVarPortReturns/100, stop = maxSrPortReturns/100)
    weights = []
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, returnTarget=target).fun)
        weights.append(((efficientOpt(meanReturns, covMatrix, returnTarget=target).x)*100).round(2))
        
    return maxSrPortReturns, maxSrPortStd, maxSrAllocation, minVarPortReturns, minVarPortStd, minVarAllocation, efficientList, targetReturns*100, weights

def EF_graph(meanReturns, covMatrix, riskFreeRate=0.05, sim=25, constraintSet=(0,1)):
    """Return a graph ploting the min vol, max sr and efficient frontier"""
    import plotly.graph_objects as go
    
    maxSrPortReturns, maxSrPortStd, maxSrAllocation, minVarPortReturns, minVarPortStd, minVarAllocation, efficientList, targetReturns = calculatedResults(meanReturns=meanReturns, covMatrix=covMatrix, riskFreeRate=riskFreeRate, sim=sim)

    #Max SR
    MaxSharpeRatio = go.Scatter(
        name='Maximium Sharpe Ratio',
        mode='markers',
        x=[maxSrPortStd],
        y=[maxSrPortReturns],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))
    )

    #Min Vol
    MinVol = go.Scatter(
        name='Mininium Volatility',
        mode='markers',
        x=[minVarPortStd],
        y=[minVarPortReturns],
        marker=dict(color='green',size=14,line=dict(width=3, color='black'))
    )

    #Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100, 2) for ef_std in efficientList],
        y=[round(target, 2) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )
    
    data = [MaxSharpeRatio, MinVol, EF_curve]

    layout = go.Layout(
        title = 'Portfolio Optimisation with the Efficient Frontier',
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Volatility (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)
    
    fig = go.Figure(data=data, layout=layout)
    return fig.show()

def EF_graphColor(assets, meanReturns, covMatrix, riskFreeRate=0.05, sim=25, constraintSet=(0, 1), export_html=False, filename='markowitzFrontier.html', figsize=(800, 600)):
    import plotly.graph_objects as go
    import plotly.io as pio
    
    maxSrPortReturns, maxSrPortStd, maxSrAllocation, minVarPortReturns, minVarPortStd, minVarAllocation, efficientList, targetReturns, weights = calculatedResults(meanReturns=meanReturns, covMatrix=covMatrix, riskFreeRate=riskFreeRate, sim=sim)

    # Calculate Sharpe ratio for each point on the efficient frontier
    sharpe_ratios = list(((targetReturns - (riskFreeRate * 100)) / efficientList) / 100)

    # Determine the color scale for EF_scatter
    color_scale = [(sharpe - min(sharpe_ratios)) / (max(sharpe_ratios) - min(sharpe_ratios)) for sharpe in sharpe_ratios]
    colors = [f'rgb(0, {int(255 * (1 - scale))}, {int(255 * scale)}' for scale in color_scale]

    # Limit the number of color bar ticks to 6
    num_ticks = 6
    tick_indices = [int(i * (len(sharpe_ratios) - 1) / (num_ticks - 1)) for i in range(num_ticks)]
    tickvals = [sharpe_ratios[i] for i in tick_indices]
    ticktext = [round(val, 2) for val in tickvals]

    # Create text for each point showing Sharpe ratio and weights with asset names
    hover_text = [f'Sharpe Ratio: {round(sharpe_ratios[i], 2)}<br>Weights:<br>' + '<br>'.join([f'{assets[j]}: {weights[i][j]}' for j in range(len(weights[i]))]) for i in range(len(efficientList))]

    EF_scatter = go.Scatter(
        name='Efficient Frontier',
        mode='markers',
        x=[round(ef_std * 100, 2) for ef_std in efficientList],
        y=[round(target, 2) for target in targetReturns],
        hovertext=hover_text,  # Use hovertext to display custom text
        marker=dict(color=sharpe_ratios, colorscale='Viridis', size=14, line=dict(width=3, color='black'),
                    colorbar=dict(title='Sharpe Ratio', tickvals=tickvals, ticktext=ticktext),
                    ))

    data = [EF_scatter]

    layout = go.Layout(
        title='Portfolio Optimization with the Efficient Frontier',
        yaxis=dict(title='Annualized Return (%)'),
        xaxis=dict(title='Annualized Volatility (%)'),
        showlegend=True,
        legend=dict(
            x=0.75, y=0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=figsize[0],  # Set width from the figsize parameter
        height=figsize[1],  # Set height from the figsize parameter
    )

    fig = go.Figure(data=data, layout=layout)
    
    if export_html:
        return pio.write_html(fig, file=filename)
    else:
        return fig

def mcEfficientFrontier(assetsList, meanReturns, covMatrix, riskFreeRate=0.05, iters=25000, figsize=(800,600), export_html=False, filename='mcEfficientFrontier.html'):
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    
    returnsList, stdList, weightsList = simulatePortfolioPerformance(meanReturns=meanReturns, covMatrix=covMatrix, n=len(assetsList), iters=iters)
    
    sharpe = (np.array(returnsList) - riskFreeRate)/np.array(stdList)
    sharpe = list(sharpe)

    hover_texts = [
        f"Sharpe Ratio: {sharpe:.2f}<br><br>" +
        "<br>".join(f"{asset}: {weight:.2%}" for asset, weight in zip(assetsList, weights))
        for sharpe, weights in zip(sharpe, weightsList)
    ]

    scatter_data = pd.DataFrame({
        'Returns': [round(returns*100,2) for returns in returnsList],
        'Standard Deviation': [round(std*100,2) for std in stdList],
        'Sharpe Ratio': sharpe,
        'Weights': weightsList
    })

    trace = go.Scatter(
        x=scatter_data['Standard Deviation'],
        y=scatter_data['Returns'],
        mode='markers',
        text=hover_texts,
        marker=dict(
            size=10,
            color=scatter_data['Sharpe Ratio'],
            colorscale='Viridis',
            colorbar=dict(title="Sharpe Ratio", ticks="outside"),
        ),
    )

    layout = go.Layout(
        title='Portfolio Scatterplot',
        xaxis=dict(title='Standard Deviation (%)'),
        yaxis=dict(title='Returns (%)'),
        width=1000,
        height=1000
    )

    fig = go.Figure(data=[trace], layout=layout)

    if export_html:
        return pio.write_html(fig, file=filename)
    else:
        return fig

def negativeDiversification(weights, stdReturns, covMatrix):
    return -(np.dot(weights,stdReturns)/np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))))

def maxDiversification(stdReturns, covMatrix, constraintSet=(0,1)):
    """maximize diversification ratio by minimizing negative diversification ratio"""
    numAssets = len(stdReturns)
    args = (stdReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(negativeDiversification, numAssets*[1./numAssets], args=args,
                                 method = 'SLSQP', bounds=bounds, constraints=constraints)
    return result

def decorrelation(weights, corrMatrix):
    return np.dot(weights.T,np.dot(corrMatrix, weights))

def maxDecorrelation(corrMatrix, constraintSet=(0,1)):
    """Maximize Decorrelation Among Assets"""
    numAssets = len(corrMatrix)
    args = (corrMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(decorrelation, numAssets*[1./numAssets], args=args,
                                 method = 'SLSQP', bounds=bounds, constraints=constraints)
    return result

def invVolatility(stdReturns):
    summ = 0
    for std in stdReturns:
        summ += 1/std
    return (1/stdReturns)/summ

def riskParity0(weights, covMatrix):
    log_weights = np.log(weights)
    log_weights[np.isinf(log_weights)] = np.nan  
    log_weights = np.nan_to_num(log_weights, nan=0.0)  
    return ((0.5 * np.dot(weights.T, np.dot(covMatrix, weights))) - ((1 / len(log_weights)) * log_weights.sum())).item()

def minEqualRisk0(covMatrix, constraintSet=(0,1)):
    """Return Weightings SUch that Each Asset Has Equal Risk Contribution to the Portfolio"""
    numAssets = len(covMatrix)
    args = (covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(riskParity0, numAssets*[1./numAssets], args=args,
                                 method = 'SLSQP', bounds=bounds, constraints=constraints)
    return result


def calculate_portfolio_var(weights, covMatrix):
    return np.dot(np.dot(weights, covMatrix), weights)

def calculate_risk_contribution(weights, covMatrix):
    sigma = np.sqrt(calculate_portfolio_var(weights, covMatrix))
    MRC = np.dot(covMatrix, weights)
    RC = np.multiply(MRC, weights) / sigma
    return RC

def risk_budget_objective(x, pars):
    # calculate portfolio risk
    covMatrix = pars[0]  
    x_t = pars[1]  
    sig_p = np.sqrt(calculate_portfolio_var(x, covMatrix))  
    risk_target = np.multiply(sig_p, x_t)
    asset_RC = calculate_risk_contribution(x, covMatrix)
    J = np.sum(np.square(asset_RC - risk_target))  
    return J

def total_weight_constraint(x):
    return 1.0 - np.sum(x)

def long_only_constraint(x):
    return x

def minEqualRisk(covMatrix):
    numAssets = len(covMatrix)
    covMatrix = np.array(covMatrix)
    x_t = np.ones(numAssets) * (1 / numAssets)  
    x_t = np.array(x_t)
    initial_guess = np.array([1 / numAssets] * numAssets)

    cons = [{'type': 'ineq', 'fun': total_weight_constraint},
            {'type': 'ineq', 'fun': long_only_constraint}]

    result = sc.optimize.minimize(risk_budget_objective, initial_guess, args=[covMatrix, x_t], method='COBYLA', constraints=cons, options={'disp': True, 'maxiter': 10000})

    # Post-process to ensure non-negative weights within [0, 1]
    weights = np.clip(result.x, 0, 1)

    # Normalize the weights to ensure they sum to 1
    weights /= weights.sum()

    return [float("{:.1e}".format(result.fun)), weights]

def plotClusters(returns, correlation = 'pearson', riskMeasure = 'vol', riskFreeRate=0.05, linkage = 'ward', max_k = 10, leafOrder=True, dendrogram=True):
    ax = rp.plot_clusters(returns=returns,
                      codependence=correlation,
                      linkage=linkage,
                      k=None,
                      max_k=max_k,
                      leaf_order=leafOrder,
                      dendrogram=dendrogram,
                      #linecolor='tab:purple',
                      ax=None)

def hierPortfolio(returns, correlation = 'pearson', riskMeasure = 'vol', riskFreeRate=0.05, linkage = 'ward', max_k = 10, leafOrder=True):    
    port = rp.HCPortfolio(returns=returns)
    
    # https://nbviewer.org/github/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%2027%20-%20HERC%20with%20Equal%20Weights%20within%20Clusters%20(HERC2).ipynb

    codependence = correlation # Correlation matrix used to group assets in clusters
    rm = riskMeasure # Risk measure used, this time will be variance
    rf = riskFreeRate # Risk free rate
    linkage = linkage # Linkage method used to build clusters
    max_k = max_k # Max number of clusters used in two difference gap statistic
    leaf_order = leafOrder # Consider optimal order of leafs in dendrogram

    w1 = port.optimization(model='HERC',
                           codependence=codependence,
                           covariance='hist',
                           rm=rm,
                           rf=rf,
                           linkage=linkage,
                           max_k=max_k,
                           leaf_order=leaf_order)

    w2 = port.optimization(model='HERC2',
                           codependence=codependence,
                           covariance='hist',
                           rm=rm,
                           rf=rf,
                           linkage=linkage,
                           max_k=max_k,
                           leaf_order=leaf_order)

    w = pd.concat([np.round(w1*100,2), np.round(w2*100, 2)], axis=1)
    w.columns = ['HERC Allocation (%)', 'HERC2 Allocation (%)']
    
    return [w, np.round(w1*100,2), np.round(w2*100, 2)]


def optimalPortfolio(method = 'maxsharpe', assetList = None, start='2017-01-01', end='2023-10-20', riskFreeRate=0.05):
    """Return Optimal Portfolio Weightings for given Asset List using Selected Method.
    returns 'x, y' optimized function value and weightings pandas DataFrame
    method = 'str' choose among ['maxsharpe', 'minstd', 'maxdiv', 'maxdecorr', 'invstd', 'eqrisk']
    assetList = 'list' of assets with full ticker including exchange suffix
    riskFreeRate = 'float' given percentage in decimal form e.g., 7% as riskFreeRate=0.07; default: 0.05
    start, end = 'str' default: start='2017-01-01', end='2023-10-20'
    """
    
    returns, meanReturns, stdReturns, covMatrix, corrMatrix = getData(assetList, start=start, end=end)
    
    if method == 'maxsharpe':
        return pd.DataFrame([np.round((portfolioPerformance(maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=riskFreeRate).x, meanReturns, covMatrix)[0])*100, 2), np.round((portfolioPerformance(maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=riskFreeRate).x, meanReturns, covMatrix)[1])*100, 2), (np.round((-maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=riskFreeRate).fun), 3))], columns = ['Value'], index = ['Returns', 'Std Dev', 'Sharpe Ratio']), pd.DataFrame((np.round((maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=riskFreeRate).x)*100, 2)), columns=['Max Sharpe Allocation (%)'], index=assetList)
    elif method == 'minstd':
        return pd.DataFrame([np.round((portfolioPerformance(minVariance(meanReturns, covMatrix).x, meanReturns, covMatrix)[0])*100, 2), np.round((portfolioPerformance(minVariance(meanReturns, covMatrix).x, meanReturns, covMatrix)[1])*100, 2)], columns = ['Value'], index = ['Returns (%)', 'Std Dev (%)']), pd.DataFrame((np.round((minVariance(meanReturns, covMatrix).x)*100, 2)), columns=['Min Std Allocation (%)'], index=assetList)
    elif method == 'maxdiv':
        return pd.DataFrame([np.round((portfolioPerformance(maxDiversification(stdReturns, covMatrix).x, meanReturns, covMatrix)[0])*100, 2), np.round((portfolioPerformance(maxDiversification(stdReturns, covMatrix).x, meanReturns, covMatrix)[1])*100, 2), -(np.round((maxDiversification(stdReturns, covMatrix).fun), 3))], columns = ['Value'], index = ['Returns (%)', 'Std Dev (%)', 'Diversification Ratio']), pd.DataFrame((np.round((maxDiversification(stdReturns, covMatrix).x)*100, 2)), columns=['Max Div Allocation (%)'], index=assetList)
    elif method == 'maxdecorr':
        return pd.DataFrame([np.round((portfolioPerformance(maxDecorrelation(corrMatrix).x, meanReturns, covMatrix)[0])*100, 2), np.round((portfolioPerformance(maxDecorrelation(corrMatrix).x, meanReturns, covMatrix)[1])*100, 2), -(np.round((maxDecorrelation(corrMatrix).fun), 3))], columns = ['Value'], index = ['Returns (%)', 'Std Dev (%)', 'Correlation']), pd.DataFrame((np.round((maxDecorrelation(corrMatrix).x)*100, 2)), columns=['Max Decorr Allocation (%)'], index=assetList)
    elif method == 'invstd':
        return pd.DataFrame([np.round((portfolioPerformance(np.array(invVolatility(stdReturns)), meanReturns, covMatrix)[0])*100, 2), np.round((portfolioPerformance(np.array(invVolatility(stdReturns)), meanReturns, covMatrix)[1])*100, 2)], columns = ['Value'], index = ['Returns (%)', 'Std Dev (%)']), pd.DataFrame((np.round((np.array(invVolatility(stdReturns)))*100, 2)), columns=['Inv Std Allocation (%)'], index=assetList)
    elif method == 'eqrisk':
        return pd.DataFrame([(np.round((portfolioPerformance(minEqualRisk(covMatrix)[1], meanReturns, covMatrix)[0]), 2)), (np.round((portfolioPerformance(minEqualRisk(covMatrix)[1], meanReturns, covMatrix)[1])*100, 2))/100, (minEqualRisk(covMatrix)[0])], columns = ['Value'], index = ['Returns (%)', 'Std Dev (%)', 'Function Value']), pd.DataFrame((np.round((minEqualRisk(covMatrix)[1])*100, 2))/100, columns=['Eq Risk Allocation (%)'], index=assetList)
    elif method == 'hier':
        plotClusters(returns)
        return pd.DataFrame([np.round(portfolioPerformance(hierPortfolio(returns, riskFreeRate=riskFreeRate)[1].values.flatten(), meanReturns, covMatrix),2), np.round(portfolioPerformance(hierPortfolio(returns, riskFreeRate=riskFreeRate)[2].values.flatten(), meanReturns, covMatrix),2)], columns = ['HERC', 'HERC2'], index = ['Returns (%)', 'Std Dev (%)']), hierPortfolio(returns, riskFreeRate=riskFreeRate)[0]
    else:
        return 'Choose a Valid Optimization Method'


def exportPortfolio(stats, weights, filename='output.xlsx', sheetsnames = ['Sheet1', 'Sheet2']):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:

        stats.to_excel(writer, sheet_name=sheetsnames[0], index=True)
        weights.to_excel(writer, sheet_name=sheetsnames[1], index=True)

st. set_page_config(layout="wide")
st.title('Portfolio Optimization')

with open('onlyTickers.json') as json_file:
    onlyTickers = json.load(json_file)

start = st.date_input("Enter start date",max_value=(dt.datetime.today() - dt.timedelta(30)), value=dt.datetime(year=2017,month=1,day=1))
end = st.date_input("Enter end date",max_value=(dt.datetime.today()), value=(dt.datetime.today()))

start = start.strftime('%Y-%m-%d')
end = end.strftime('%Y-%m-%d')

stocks = st.multiselect(label='Select Stocks', options=onlyTickers)

riskFreeRate = st.number_input(label='Enter Risk Free Rate in decimal (Default 0.05 i.e, 5%)', min_value=0.00, max_value=1.00, value=0.05, step=0.001, format="%0.4f")

st.text('Optimizations Available:')
st.text('Maximize Sharpe Ratio: maxsharpe')
st.text('Minimize Volatility: minstd')
st.text('Maximize Diversification: maxdiv')
st.text('Maximize Decorrelation: maxdecorr')
st.text('Inverse Volatility: invstd')
st.text('Hierarchical Equal Risk Contribution (dev): eqrisk')
st.text('Note: maxsharpe and minstd allows Download of Markowitz Efficient Frontier')

method = st.selectbox(label="Select Optimization Technique", options=['maxsharpe', 'minstd', 'maxdiv', 'maxdecorr', 'invstd', 'eqrisk'])

cont=False

if start and end and stocks and riskFreeRate and method:

    submit_button = st.button("Submit")

    if submit_button:

        returns, meanReturns, stdReturns, covMatrix, corrMatrix = getData(stocks, start=start, end=end)

        st.text('Loading data...done!')

        stats, weights = optimalPortfolio(method=method,assetList=stocks)

        st.write(stats, weights)

        st.text('Computing... done!')
        cont=True


def save_plot_as_image(fig, filename):
    fig.write_image(filename, format="png")

# Define a function to convert an image to PDF
def convert_image_to_pdf(image_filename, pdf_filename):
    c = canvas.Canvas(pdf_filename)
    c.drawImage(image_filename, 100, 100)
    c.save()

if (method == 'maxsharpe' or method == 'minstd') and cont:
    try:
        # Create a Plotly figure using your EF_graphColor function
        fig = EF_graphColor(stocks, meanReturns, covMatrix, riskFreeRate)

        # Save the Plotly figure as an HTML file
        html_filename = "efront.html"
        fig.write_html(html_filename)

        # Provide a download button for the HTML file
        st.text('Download the Efficient Frontier as an HTML file:')
        st.download_button(
            label='Download HTML',
            data=open(html_filename, 'rb').read(),
            key='eFront_html',
            file_name='eFront.html',
        )
    except Exception as e:
        st.error(f"Error: {e}")

st.text('References')
references = [
    "Haugen, R. and Baker, N. (1991) 'The Efficient Market Inefficiency of Capitalization-Weighted Stock Portfolios.' Journal of Portfolio Management, 17, 35-40."
    "Choueifaty, Yves, and Yves Coignard. 2008. 'Toward Maximum Diversification.' Journal of Portfolio Management Vol 35 Issue 1.",
    "Choueifaty, Yves, Tristan Froidure, and Julien Reynier. 2012. 'Properties of the Most Diversified Portfolio.' Journal of Investment Strategies Volume 2 Issue 2.",
    "DeMiguel, Victor, Lorenzo Garlappi, and Raman Uppal. 2007. 'Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?' Oxford University Press.",
    "Christoffersen, P., V. Errunza, K. Jacobs, and X. Jin. 2010. 'Is the Potential for International Diversification Disappearing?'",
    "Haugen, R., and N. Baker. 1991. 'The Efficient Market Inefficiency of Capitalization-Weighted Stock Portfolios.' Journal of Portfolio Management 17.",
    "Lopez de Prado, Marcos. 2016. 'Building Diversified Portfolios that Outperform Out of Sample.' Journal of Portfolio Management 42 (4): 59–69.",
    "Maillard, Sebastien, Thierry Roncalli, and Jerome Teiletche. 2008. 'On the properties of equally-weighted risk contributions portfolios.'",
    "Poterba, James M., and Lawrence H. Summers. 1988. 'Mean Reversion in Stock Prices: Evidence and Implications.' Journal of Financial Economics 22 (1).",
    "Spinu, Florin. 2013. 'An Algorithm for Computing Risk Parity Weights.'"
]

for reference in references:
    st.text(reference)

if st.button("Report Bug"):
    st.subheader("Bug Report Form")
    bug_description = st.text_area("Describe the bug:")
    bug_button = st.button("Submit Bug Report")

    if bug_button:
        # Store the bug report in a text file
        with open("bug_reports.txt", "a") as file:
            file.write(bug_description + "\n")
        st.success("Bug report submitted. Thank you!")






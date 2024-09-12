import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
import time

start_time = time.time()
data=pd.read_excel('50stocks.xlsx')
returns=data.mean()*252
cov_matrix=data.cov()*252
rfr=0.03

def portfolioReturn(weights):
    return np.sum(weights*returns)
def portfolioVolatility(weights):
    return np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))

numAssets=len(data.columns)
minRet=min(returns)
maxRet=max(returns)
trets=np.linspace(minRet,maxRet,200)
tvols=[]
weights_max_sharpe=None
returns_max_sharpe=None
volatility_max_sharpe=None
sharpe_ratio_max_sharpe=-float("inf")
initialWeights=np.ones(numAssets)/numAssets
bnds=tuple((0,1) for _ in range(numAssets))

for tret in trets:
    cons=({"type":"eq","fun":lambda x:portfolioReturn(x)-tret},
          {"type":"eq","fun":lambda x:np.sum(x)-1}
    )
    res=sco.minimize(portfolioVolatility,initialWeights,method='SLSQP',bounds=bnds,constraints=cons)
    if res.success:
        frontierWeights=res["x"]
        frontierRet=portfolioReturn(frontierWeights)
        frontierVol=portfolioVolatility(frontierWeights)
        tvols.append(res["fun"])
        sharpe_ratio=(frontierRet-rfr)/frontierVol
        if sharpe_ratio>sharpe_ratio_max_sharpe:
            sharpe_ratio_max_sharpe=sharpe_ratio
            weights_max_sharpe=frontierWeights
            returns_max_sharpe=frontierRet
            volatility_max_sharpe=frontierVol

print("expected returns:",returns_max_sharpe)
print("expected volatility:",volatility_max_sharpe)
print("max sharpe ratio:",sharpe_ratio_max_sharpe)
print("weights:",weights_max_sharpe)

plt.figure(figsize=(16,8))
plt.scatter(tvols,trets,c=(np.array(tvols)-min(tvols))/(max(tvols)-min(tvols)),marker='+',cmap='viridis')
plt.scatter(volatility_max_sharpe,returns_max_sharpe,marker="*",color="r",s=300,label="max sharpe ratio")
plt.grid(True)
plt.xlabel("expected volatility")
plt.ylabel("expected return")
plt.colorbar(label="expected volatility")
plt.legend()
plt.savefig("Standard efficient frontier corresponding to SSE50 problem.png")
plt.show()

end_time = time.time()
print(f'Time taken: {end_time - start_time} seconds')
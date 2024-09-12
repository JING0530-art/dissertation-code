import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import scipy.optimize as sco
import matplotlib as mpl

start_time = time.time()
data=pd.read_excel('50stocks.xlsx')
returns=data.mean()*252
cov_matrix=data.cov()*252


def portfolioReturn(weights):
    return np.sum(weights*returns)
def portfolioVolatility(weights):
    return np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))

def portfolio_fitness(weights, returns,risk_free_rate):
    """
    Calculates the fitness of a portfolio given the weights and returns.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    portfolio_sharpe_ratio = (portfolio_return-risk_free_rate) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, portfolio_sharpe_ratio])


def particle_swarm_optimization(returns, num_particles, max_iterations, alpha, beta, risk_free_rate):
    """
    Performs portfolio optimization using particle swarm optimization.
    """
    # Initialize particles and velocities
    particles = np.random.rand(num_particles, returns.shape[1])
    velocities = np.zeros((num_particles, returns.shape[1]))

    # Initialize global best
    global_best_fitness = None  # Array
    global_best_weights = None  # Array

    xaxis=[]
    yaxis=[]
    sharpe_ratios=[]

    # Iterate through each particle
    for i in range(num_particles):

        particles[i] = particles[i] / np.sum(particles[i])
        # Calculate particle fitness
        particle_fitness = portfolio_fitness(particles[i], returns, risk_free_rate)

        # Initialize particle best
        particle_best_fitness = particle_fitness
        particle_best_weights = particles[i]

        # Update global best
        if global_best_fitness is None or particle_best_fitness[2] > global_best_fitness[2]:
            global_best_fitness = particle_best_fitness
            global_best_weights = particle_best_weights

        # Iterate through iterations
        for j in range(max_iterations):
            # Update velocity
            velocities[i] = alpha * velocities[i] + beta * np.random.uniform(0, 1) * (
                        particle_best_weights - particles[i]) + beta * np.random.uniform(0, 1) * (
                                        global_best_weights - particles[i])

            # Update particle position
            particles[i] = particles[i] + velocities[i]

            particles[i] = particles[i] / np.sum(particles[i])
            # Ensure particle weights are within bounds
            particles[i] = np.minimum(particles[i], 1)
            particles[i] = np.maximum(particles[i], 0)

            # Calculate particle fitness
            particle_fitness = portfolio_fitness(particles[i], returns, risk_free_rate)

            # Update particle best
            if particle_fitness[2] > particle_best_fitness[2]:
                particle_best_fitness = particle_fitness
                particle_best_weights = particles[i]

                # Update global best
                if particle_best_fitness[2] > global_best_fitness[2]:
                    global_best_fitness = particle_best_fitness
                    global_best_weights = particle_best_weights

            xaxis.append(particle_fitness[1])
            yaxis.append(particle_fitness[0])
            sharpe_ratios.append(particle_fitness[2])

    return np.array(xaxis), np.array(yaxis),np.array(sharpe_ratios),global_best_fitness,global_best_weights



"""
def analytical_optimization(returns,cov_matrix,risk_free_rate):
    num_assets=len(returns)

    def neg_sharpe_ratio(weights,returns,cov_matrix,risk_free_rate):
        p_ret=np.sum(returns * weights) * 252
        p_vol=np.sqrt(np.dot(weights.T,np.dot(cov_matrix*252,weights)))
        return -(p_ret-risk_free_rate)/p_vol

    constraints=({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bounds=tuple((0,1) for asset in range(num_assets))
    initial_weights=num_assets*[1./num_assets]

    opts=sco.minimize(neg_sharpe_ratio,initial_weights,args=(returns,cov_matrix,risk_free_rate),method="SLSQP",constraints=constraints,bounds=bounds)
    return opts["x"],opts
"""

numAssets=len(data.columns)
minRet=min(returns)
maxRet=max(returns)
trets=np.linspace(minRet,maxRet,100)
tvols=[]
weights_max_sharpe=None
returns_max_sharpe=None
volatility_max_sharpe=None
sharpe_ratio_max_sharpe=-float("inf")
initialWeights=np.ones(numAssets)/numAssets
bnds=tuple((0,1) for _ in range(numAssets))
risk_free_rate=0.03

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
        sharpe_ratio=(frontierRet-risk_free_rate)/frontierVol
        if sharpe_ratio>sharpe_ratio_max_sharpe:
            sharpe_ratio_max_sharpe=sharpe_ratio
            weights_max_sharpe=frontierWeights
            returns_max_sharpe=frontierRet
            volatility_max_sharpe=frontierVol

# Measure time
num_particles=500
max_iterations=500
alpha=0.9
beta=0.6
xaxis,yaxis,sharpe_ratios,pso_fitness,pso_weights=particle_swarm_optimization(data,num_particles,max_iterations,alpha,beta,risk_free_rate)
#analytical_weights,analytical_fitness=analytical_optimization(returns,cov_matrix,rfr)
#euclidean_distance=np.linalg.norm(pso_weights-analytical_weights)
print("maxweights_pso")
print(f"expected return:{pso_fitness[0]}")
print(f"expected volatility:{pso_fitness[1]}")
print(f"expected sharpe_ratio:{pso_fitness[2]}")
print(f"weights:{pso_weights}")

'''
print(f"_weights:{pso_weights}")
print(f"_return:{pso_fitness[0]}")
print(f"_volatility:{pso_fitness[1]}")
print(f"_sharpe:{pso_fitness[2]}")
'''

"""
print("-------------------------")
print("maxweights_analytical")
print(f"_weights:{analytical_weights}")
print(f"_return:{np.sum(analytical_weights*returns)*252}")
print(f"_volatility:{np.sqrt(np.dot(analytical_weights.T,np.dot(cov_matrix*252,analytical_weights)))}")
print(f"_sharpe:{-analytical_fitness["fun"]}")
"""

plt.figure(figsize=(16,10),layout="constrained")
plt.scatter(xaxis,yaxis,c=sharpe_ratios,cmap=mpl.cm.viridis,marker="o",label="PSO")
plt.plot(tvols,trets,color="blue",marker=".",label="Analytical Frontier")
#plt.scatter(pso_fitness[1],pso_fitness[0],marker="*",color="green",s=300,label="max_pso")
plt.scatter(volatility_max_sharpe,returns_max_sharpe,marker="*",color="red",s=300,label="max_analytical")
plt.grid(True)
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.legend()
plt.savefig("1.png")
plt.show()
end_time = time.time()

print(f'Time taken: {end_time - start_time} seconds')
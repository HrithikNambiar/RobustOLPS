from random import random
from random import uniform
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
# We will meta-learn weights on these algos
from universal_olps.universal.algos import BAH, CRP, BCRP, DynamicCRP, UP, EG, Anticor, PAMR, OLMAR, RMR, CWMR, WMAMR, RPRT, BNN, CORN, BestMarkowitz, Kelly, BestSoFar, ONS
from universal_olps.universal.result import ListResult
import pandas as pd

"""
Idea weight apply a weight vector [w1,w2,w3,w4,w5,w6,w7,w8] to these algorithms/experts.
We do this using PSO. 
"""

class Particle:
    def __init__(self, x0, algo_assetWeights, algo_data):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.algo_assetWeights = algo_assetWeights
        self.algo_data = algo_data

        for i in range(0,num_dimensions):
            self.velocity_i.append(uniform(-0.1,0.1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i,self.algo_assetWeights,self.algo_data)

        # check to see if the current position is an individual best
        if self.err_i>self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=self.err_i
                    
    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant
        
        for i in range(0,num_dimensions):
            r1=random()
            r2=random()
            
            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            # Weights should be between 0 and 1 
            
            # adjust maximum position if necessary
            if self.position_i[i]>bounds[1]:
                self.position_i[i]=bounds[1]

            # adjust minimum position if neseccary
            if self.position_i[i]<bounds[0]:
                self.position_i[i]=bounds[0]
        

"""
The goal is to MAXIMIZE WEALTH!
"""
def maximize_wealth(costFunc, initial_particles, algo_assetWeights, algo_data, bounds, num_particles, maxiter, verbose=False):
    global num_dimensions
    position_across_iterations = []
    num_dimensions = initial_particles.shape[1]
    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group

    # establish the swarm
    swarm=[]
    for i in range(0,num_particles):
        swarm.append(Particle(initial_particles[i], algo_assetWeights, algo_data))

    # begin optimization loop
    i=0
    while i<maxiter:
        if verbose: print(f'iter: {i:>4d}, best solution: {err_best_g:10.6f}')

        # cycle through particles in swarm and evaluate fitness
        for j in range(0,num_particles):
            swarm[j].evaluate(costFunc)

            # determine if current particle is the best (globally)
            if swarm[j].err_i>err_best_g or err_best_g==-1:
                pos_best_g=list(swarm[j].position_i)
                err_best_g=float(swarm[j].err_i)
        
        # cycle through swarm and update velocities and position
        for j in range(0,num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)
        i+=1
        position_across_iterations.append(pos_best_g)
    # print final results
    # if err_best_g==1.0:
    #     print("equity is 1.0")
    #     print("algo data: {}".format(algo_data))
    if verbose:
        print('\nFINAL SOLUTION:')
        print(f'   > {pos_best_g}')
        print(f'   > {err_best_g}\n')

    return err_best_g, pos_best_g,position_across_iterations,swarm


"""
Cost function -> Wealth / Sharpe ratio
"""
def fitness_fn(weights,algo_assetWeights,algo_data):
    # return the total wealth/equity gained
    # weights = softmax(weights)
    weights = np.array(weights)
    weights = weights/sum(weights)
    weighted_assetWeights = None
    stock_data = None
    for i,key in enumerate(algo_assetWeights):
        if i==0: 
            weighted_assetWeights = weights[0]*algo_assetWeights[key]
            stock_data = algo_data[key]
            # stock_data = stock_data[stock_data.index<train_cutoff]
        weighted_assetWeights += weights[i]*algo_assetWeights[key]
    # weighted_assetWeights = weighted_assetWeights[weighted_assetWeights.index<train_cutoff]
    equity = np.maximum(((np.array(stock_data)-1) * weighted_assetWeights).sum(axis=1)+1, 1e-10).cumprod()
    #print("The weights are {} and equity is {}".format(weights,np.array(equity)[-1]))
    # print("Equity: {}".format(np.array(equity)))
    if np.array(equity)[-1]==1.0:
        print("equity is 1.0")
        print("returns {}".format(np.maximum(((np.array(stock_data)-1) * weighted_assetWeights).sum(axis=1)+1, 1e-10)))
        print("equity {}".format(equity))
    return np.array(equity)[-1]

def fitness_fn_modified(weights,algo_assetWeights,algo_data,start_date, end_date):
    # return the total wealth/equity gained
    # weights = softmax(weights)
    weights = np.array(weights)
    weights = weights/sum(weights)
    weighted_assetWeights = None
    stock_data = None
    for i,key in enumerate(algo_assetWeights):
        if i==0: 
            weighted_assetWeights = weights[0]*algo_assetWeights[key]
            stock_data = algo_data[key]
            stock_data = stock_data.loc[start_date:end_date]
        weighted_assetWeights += weights[i]*algo_assetWeights[key]
    
    weighted_assetWeights = weighted_assetWeights.loc[start_date:end_date]
    equity = np.maximum(((np.array(stock_data)-1) * weighted_assetWeights).sum(axis=1)+1, 1e-10).cumprod()
    # print("The weights are {} and equity is {}".format(weights,np.array(equity)[-1]))
    return np.maximum(((np.array(stock_data)-1) * weighted_assetWeights).sum(axis=1)+1, 1e-10), np.array(equity)[-1]


"""
Run all olps algo and save their weights
"""
results = []
algo_assetWeights = {}
algo_data = {}
olpss = [UP(), EG(), PAMR(), OLMAR(), RMR(), CWMR(), CORN(), ONS()]
algo_names = [algo.__class__.__name__ for algo in olpss]
train_cutoff = pd.to_datetime("2020-01-01").tz_localize("UTC")
for algo in olpss:
    try:
        run_res = algo.run(S)
        results.append(run_res)
        algo_assetWeights[algo.__class__.__name__] = run_res.weights
        algo_data[algo.__class__.__name__] = run_res.X
    except Exception as e:
        print(f"Could not run {algo.__class__.__name__}: {e}")
        
df = ListResult(results, [algo.__class__.__name__ for algo in olpss]).to_dataframe()
train_df = df.reset_index(drop=False)
train_df = train_df[train_df['Date'] < train_cutoff]

def main():
    olpss = [UP(), EG(), PAMR(), OLMAR(), RMR(), CWMR(), CORN(), ONS()]
    num_particles = 200
    num_strats = 8
    initial_particles = np.random.uniform(0,1,(num_particles,num_strats))
    bounds=[0,1]
    err_best_g, pos_best_g,position_across_iterations,swarm = maximize_wealth(fitness_fn, initial_particles, algo_assetWeights, algo_data, bounds, num_particles, maxiter=40, verbose=True)
    weights_to_strats = position_across_iterations
    plt.plot(weights_to_strats[0:10])
    plt.legend([algo.__class__.__name__ for algo in olpss])

    plt.xlabel('PSO Iterations')
    plt.ylabel('Weight to Strategies')
    plt.title("Evolutions of weights to strategies over PSO iterations (Trained 2010-2020)")
    plt.show()

if __name__ == "__main__":
    main()

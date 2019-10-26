# posterior inference by gibbs sampling
import numpy as np
import pymc3 as pm

x = np.array([[0.52, 1.12,  0.77],
               [0.88, -1.08, 0.15],
               [0.52, 0.06, -1.30],
               [0.74, -2.49, 1.39],
               [0.52, 1.12,  0.77]])

y = np.array([True, True, False, True, False])

data = {'x0': x[:,0], 'x1': x[:,1], 'x2': x[:,2], 'y': y}

with pm.Model() as logistic_model:
  pm.glm.GLM.from_formula('y ~ x0 + x1 + x2', data=data, family='binomial')
  trace = pm.sample(900, step=pm.NUTS())
  pp = pm.sampling.sample_posterior_predictive(trace)
  print (sum(pp['y'])/len(pp['y']))
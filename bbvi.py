# posterior inference by black-box variational inference


import autograd.numpy as np
from autograd import grad
import autograd.scipy as scipy
import autograd.scipy.stats.norm as norm

from autograd.misc.optimizers import adam, sgd

def sigmoid(x):
  return 0.5*(np.tanh(x/2)+1)

def predict(w, x):
  return sigmoid(np.dot(x, w))

def log_sigmoid(x):
  a = np.array([np.zeros_like(x), -x])
  return -scipy.special.logsumexp(a, axis=0)

def log_joint(w, x, y, alpha=0.1):
  log_prior = alpha*np.sum(w**2)

  score = np.dot(x,  w)
  logp0 = log_sigmoid(score)

  logp1 =  -score + logp0
  log_likelihood = np.sum(y*logp0  + (1-y)*logp1, axis=0)

  return log_likelihood - log_prior





def bbvi(x, y, params, log_joint, T=25):
  ''' Perform black-box variational inference
    with q = diagonal gaussian
  '''
  def objective(params, i=0):
    D = len(params) // 2
    mu, log_sigma2 =  params[:D], params[D:]

    entropy = 0.5*D*(1 + np.log(2*np.pi)) + np.sum(log_sigma2)
    
    # samples = np.random.randn(T, D) * np.exp(0.5*log_sigma2) + mu
    logp = 0
    sample = mu + np.random.randn(T, D) * np.exp(0.5*log_sigma2) 

    for t in range(T):
      logp += log_joint(sample[t], x, y)
    return -(logp/T + entropy)

  gradient = grad(objective)
  return objective, gradient




x = np.array([[0.52, 1.12,  0.77],
               [0.88, -1.08, 0.15],
               [0.52, 0.06, -1.30],
               [0.74, -2.49, 1.39]])

y = np.array([True, True, False, True])

x = np.hstack([np.ones(( len(x),1)), x])

params = np.zeros(4+4) 
objective, gradient = bbvi(x, y, params, log_joint, T=100)
print (objective(params))
print (params)
print (predict(params[:4], x))

params = adam(gradient, params, step_size=0.01, num_iters=500)

print (objective(params), objective(params))
print (np.exp(params[4:]))
print (predict(params[:4], x))
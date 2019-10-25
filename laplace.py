# posterior inference by laplace approximation
from autograd import grad
import autograd.numpy as np
import autograd.scipy as scipy


def sigmoid(x):
  return 0.5*(np.tanh(x/2)+1)

def predict(w, x):
  return sigmoid(np.dot(x, w))

def log_sigmoid(x):
  a = np.array([np.zeros_like(x), -x])
  return -scipy.special.logsumexp(a)

def nll_loss(w, x, y, alpha=None):
  score = np.dot(x, w)
  
  logp1 = log_sigmoid(score)
  logp0 = -score + logp1
  loss = y*logp0  + (1-y)*logp1
  reg = alpha*np.dot(w, w) if alpha else 0
  return sum(loss) + reg

def compute_precision(x, y, w, alpha):
  d = np.size(x, 1)
  y_hat = predict(w, x)
  R = np.diag(y_hat*(1 - y_hat))
  precision = alpha * np.eye(d) + x.T.dot(R).dot(x)
  return precision + 1e-9*np.eye(d)

def predict_mc(mu, sigma, x, T=100):
  ps = []
  for t in range(T):
    w = np.random.multivariate_normal(mu, sigma)
    ps.append(predict(w, x))
  return sum(ps) / T

def predict_var(mu, sigmainv, x):
  mu_a = np.dot(x, w)
  sigma2_a = np.sum(np.linalg.solve(sigmainv, x.T).T * x, axis=1)

  kappa = np.sqrt(1 + sigma2_a*np.pi*.125)
  return sigmoid(mu_a/kappa) 

x = np.array([[0.52, 1.12,  0.77],
               [0.88, -1.08, 0.15],
               [0.52, 0.06, -1.30],
               [0.74, -2.49, 1.39],
               [0.52, 1.12,  0.77]])
y = np.array([True, True, False, True, False])


x = np.hstack([np.ones(( len(x),1)), x])
training_loss = lambda w: nll_loss(w, x, y, alpha=1)
g = grad(training_loss)
w = np.array([1, 0, 0, 0], dtype=np.float)
print("Initial loss:", training_loss(w))
for i in range(100):
    w -= g(w) * 0.01
print("Trained loss:", training_loss(w))

pred = predict(w, x) > 0.5

print (y.astype(int))
print (predict(w, x) )


sigmainv = compute_precision(x,y,w,alpha=1)

print (predict_var(w, sigmainv, x))
print (predict_mc(w, np.linalg.inv(sigmainv), x))
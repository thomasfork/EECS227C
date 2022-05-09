import numpy as np
import cvxpy as cv
import casadi as ca
import matplotlib.pyplot as plt

n = 20
c = np.ones(n)
N = 1000000

tau = 0.29
beta = 0.126
gamma = 0.164

#x = cv.Variable(n)
#F = sum(-cv.log(x[i] + 1) for i in range(n)) + sum(-cv.log(1 - x[i]) for i in range(n))

x = ca.SX.sym('x',n)
t = ca.SX.sym('t')
F = sum(-ca.log(x[i]+1) for i in range(n)) + sum(-ca.log(1-x[i]) for i in range(n))

del_F = ca.jacobian(F,x)[0,:].T
del2_F = ca.jacobian(del_F, x)

v = ca.SX.sym('v',n)
x_norm = ca.Function('x_norm', [x,v], [ca.sqrt(v.T @ del2_F @ v)])

t_plus = t + gamma / x_norm(x,c)
lamb = x_norm(x, t_plus*c + del_F)
xi = lamb**2 / (1 + lamb)
x_plus = x - 1/(1+xi) * ca.inv(del2_F) @ (t_plus*c + del_F)

alg_step = ca.Function('step',[t,x],[t_plus, x_plus])

x0 = np.zeros(n)
t0 = 1
x_star = -np.ones(n)

err = ca.Function('err',[x],[x.T @ c + n])
err_func = lambda x: err(x).__float__()

errs = [err_func(x0)]
ts = [t0]
t = t0
x = x0.copy()

for k in range(N):
    t, x = alg_step(t,x)
    errs.append(err_func(x))
    ts.append(t.__float__())

plt.plot(range(1,N+2), errs, label = 'Suboptimality Gap')
plt.plot(range(1,N+2), ts, label = 't')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlabel('Iteration')
plt.show()

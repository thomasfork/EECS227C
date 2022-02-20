import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

K = 5
n = 10

A = np.zeros((K,n,n))
b = np.zeros((K,n))

for k in range(1,K+1):
    for j in range(1,n+1):
        for i in range(1,j):
            A[k-1,i-1,j-1] = np.exp(i/j) * np.cos(i*j) * np.sin(k)
            A[k-1,j-1,i-1] = np.exp(i/j) * np.cos(i*j) * np.sin(k)
    for i in range(1,n+1):
        b[k-1,i-1] = np.exp(i/k) * np.sin(i*k)
        A[k-1,i-1,i-1] = i/10*np.abs(np.sin(k)) \
                       + sum(np.abs(A[k-1,i-1,j-1]) if j != i else 0 
                             for j in range(1,n+1)) 

x1 = np.ones(n)

x = ca.SX.sym('x',n)
fk = ca.vertcat(*[ca.bilin(A[k],x,x)- ca.dot(b[k], x) for k in range(K)])
gk = ca.jacobian(fk, x)
f = ca.mmax(fk)

F = ca.Function('f',[x], [f])
Fk = ca.Function('f',[x], [fk])
Ik = ca.Function('I',[x], [fk == f]) # indicator functions for subgradient
Gk = ca.Function('g',[x], [gk])

z = ca.SX.sym('z')
g = ca.vertcat(*[z - fk[k] for k in range(K)])
ubg = [np.inf] * K
lbg = [0.0] * K

prob = {'x':ca.vertcat(z,x),'f':z, 'g':g}
opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}
solver = ca.nlpsol('fopt', 'ipopt', prob, opts)

f1 = F(x1)
f_opt = solver(ubg = ubg, lbg = lbg)['f']
x_opt = np.array(solver(ubg = ubg, lbg = lbg)['x'][1:]).squeeze()
print('Part A:')
print('f(x1) is %0.2f'%f1)
print('The optimal value is %0.6f'%f_opt)
print('and is achieved by x = %s'%x_opt)
print('')

def subgradient(x):
    gk_full = Gk(x)
    Ik_full = Ik(x)
    gk_set = np.array([gk_full[k,:] for k in range(K) if Ik_full[k] == 1]).squeeze()
    if gk_set.ndim == 1:
        if (gk_set == np.zeros(n)).all():
            return gk_set, True
        else:
            return gk_set, False
    else: 
        import pdb
        pdb.set_trace()

def learning_rate(C, t):
    return C / np.sqrt(t)

def learning_rate_polyak(x, g):
    return (F(x) - f_opt) / np.linalg.norm(g)

def step(x, t, C = 2, polyak = False):
    g, stop = subgradient(x)
    if stop:
        pass
    else:
        if polyak:
            lr = learning_rate_polyak(x, g)
        else:
            lr = learning_rate(C, t)
        xp1 = x - lr * g / np.linalg.norm(g)
    return xp1, F(xp1), stop

def rollout(polyak = False):
    x = x1.copy()
    J_plot = [F(x).__float__()]
    J_best = np.inf
    for k in range(10000):
        x, J, stop = step(x, k+1, polyak = polyak)
        J_best = min(J_best, J.__float__())
        J_plot.append(J_best)
    return J_plot

J_b = rollout(False)
J_c = rollout(True)

plt.figure()
plt.plot(range(1,len(J_b)+1), J_b - f_opt, label = 'Part B')
plt.plot(range(1,len(J_c)+1), J_c - f_opt, label = 'Part C')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Suboptimality Gap')
plt.xlabel('Iteration')
plt.legend()
plt.show()

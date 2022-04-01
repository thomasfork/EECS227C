import numpy as np
import cvxpy as cv
import matplotlib.pyplot as plt

K = 5
n = 10
N = 100

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


# helper functions
fj = lambda x: [x.T @ A[k] @ x - b[k].T @ x for k in range(K)]
f = lambda x: max(fj(x)) # objective value
I = lambda x: np.nonzero(fj(x) == f(x)) # indicator function
g = lambda x: 2 * A[I(x)[0]] @ x - b[I(x)[0]]  # a valid subgradient

# the exact solution
z = cv.Variable()
x = cv.Variable(n)
constraints = [z >= cv.quad_form(x,A[k]) - b[k].T @ x for k in range(K)]
prob = cv.Problem(cv.Minimize(z), constraints)
f_star = prob.solve(solver = cv.ECOS)
print(f_star)
print(x.value)




# the level method           
G = [x <= 1, x >= -1] # convex domain to bound decision variable
model_sequence = []       # sequence of function approximations
f_plus = np.inf           # best objective found so far
lam = 0.5                 # lambda term



xi = np.ones(n)
suboptimality = [f(xi) - f_star]
for i in range(N-1):
    
    fxi = f(xi)
    gxi = g(xi)
    
    model_sequence.append(z >= fxi + gxi @ (x -xi))
    
    f_minus_prob = cv.Problem(cv.Minimize(z), [*model_sequence, *G])
    f_minus = f_minus_prob.solve(solver = cv.ECOS)
    
    f_plus = min(f_plus, fxi)
    
    li = lam * f_plus + (1-lam) * f_minus
    
    x_proj_prob = cv.Problem(cv.Minimize(cv.quad_form(x-xi, np.eye(n))),
                             [*model_sequence, *G, z <= li])
    x_proj_prob.solve(solver = cv.ECOS)
    
    xi = x.value
    
    suboptimality.append(f(xi) - f_star)


plt.plot(np.arange(1,N+1), suboptimality)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Suboptimality Gap')
plt.xlabel('Iteration')
plt.title('Level Method Convergence on MAXQUAD')
plt.show()




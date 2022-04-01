import numpy as np
import cvxpy as cv
import matplotlib.pyplot as plt

n = 20
N = 3000

def exact_sol(A):
    x = cv.Variable(n)
    prob = cv.Problem(cv.Minimize(cv.quad_form(x, A.T @ A)))
    f_star = prob.solve(solver = cv.ECOS)
    x_star = x.value
    return f_star, x_star

def gd(A):
    L = np.linalg.eigh(2*A.T @ A)[0].max()
    h = 1/L
    
    f_star, _ = exact_sol(A)
    f = lambda x: x.T @ A.T @ A @ x
    
    xi = np.ones(n)
    f_best = f(xi) - f_star
    suboptimality = [f_best]
    
    for i in range(N-1):
        gi = 2 * A.T @ A @ xi
        
        xi = xi - h * gi
        f_best = min(f_best, f(xi) - f_star)
        suboptimality.append(f_best)
    
    return suboptimality

def agd(A):
    L = np.linalg.eigh(2*A.T @ A)[0].max()
    S = np.linalg.eigh(2*A.T @ A)[0].min()
    h = 4 / (np.sqrt(L) + np.sqrt(S))**2
    th = max(abs(1-np.sqrt(h*L)), abs(1-np.sqrt(h*S)))**2
    
    f_star, _ = exact_sol(A)
    f = lambda x: x.T @ A.T @ A @ x
    
    #gam = lambda i: 0 if i <= 3 else 2/i 
    #lam = lambda i: 1 if i <= 3 else 6/i/(i-1)
    
    xi_m1 = np.ones(n)
    xi = np.ones(n)
    zi = xi.copy()
    f_best = f(xi) - f_star
    suboptimality = [f_best]
    
    for i in range(N-1):
        #lam_i = lam(i)
        #gam_i = gam(i)
        
        #yi = (1-gam_i) * xi + gam_i * zi
        #gi = 2 * A.T @ A @ yi
        
        
        #xi = xi - h * gi
        #zi = zi - gam_i / lam_i * gi / np.linalg.norm(gi)
        gi = 2*A.T @ A @ xi
        xi_p1 = xi - h * gi + th * (xi - xi_m1)
        
        xi_m1 = xi.copy()
        xi = xi_p1
        
        
        f_best = min(f_best, f(xi) - f_star)
        suboptimality.append(f_best)
        
    return suboptimality


A = np.random.uniform(size = (n,n))
gd_result = gd(A)
agd_result = agd(A)

plt.figure()
plt.plot(np.arange(1,N+1), gd_result, label = 'Gradient Descent')
plt.plot(np.arange(1,N+1), agd_result, label = 'Accelerated Gradient Descent')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Suboptimality Gap')
plt.xlabel('Iteration')
plt.title('Random A Matrix')
plt.legend()


A = np.eye(n)*2 - np.eye(n,k=1) - np.eye(n,k=-1)
gd_result = gd(A)
agd_result = agd(A)

plt.figure()
plt.plot(np.arange(1,N+1), gd_result, label = 'Gradient Descent')
plt.plot(np.arange(1,N+1), agd_result, label = 'Accelerated Gradient Descent')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Suboptimality Gap')
plt.xlabel('Iteration')
plt.title('Provided A Matrix')
plt.legend()


plt.show()

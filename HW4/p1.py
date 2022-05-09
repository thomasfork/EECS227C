import casadi as ca
import matplotlib.pyplot as plt

n = 10
Mf = 1

eps = ca.SX.sym('e')
x   = ca.SX.sym('x', n)

f = 1/eps * sum(k * x[k] for k in range(n)) \
  - sum(ca.log(1-x[k]**2) for k in range(n))

d1f = ca.jacobian(f, x).T
d2f = ca.jacobian(d1f, x)

lambda_f = ca.sqrt(d1f.T @ ca.inv(d2f) @ d1f)

x_p1 = x - 1/(1+Mf * lambda_f) * ca.inv(d2f) @ d1f

f_lambda = ca.Function('lambda', [x, eps], [lambda_f])
f_step = ca.Function('step', [x, eps], [x_p1, lambda_f])

eps_list = [1, 0.1, 0.01, 0.005]
lambda_threshold = 1e-6

for eps in eps_list:
    x = [0] * n
    lambda_x = float(f_lambda(x, eps))
    lambda_list = [lambda_x]

    while lambda_x >= lambda_threshold:
        x, lambda_x = f_step(x, eps)
        lambda_list.append(float(lambda_x))
    
    plt.plot(range(1, len(lambda_list)+1), lambda_list, label = f'e = {eps}')

plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Iteration')
plt.ylabel('Lambda_f')
plt.show()


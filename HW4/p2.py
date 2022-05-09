from sympy import * 

x1, x2, d = symbols('x1 x2 d')

f = log(1/(d-x1)) \
  + log(1/(d+x1)) \
  + log(1/(1-x2)) \
  + log(1/(1+x2))
  
# hessian is diagonal so we can simplify computation of lambda:
lambda_f = sqrt(diff(f, x1)**2 / diff(f, x1, x1) + diff(f, x2)**2 / diff(f, x2, x2))

pprint(lambda_f)

# print the two halves simplified
pprint(simplify(diff(f, x1)**2 / diff(f, x1, x1)))
pprint(simplify(diff(f, x2)**2 / diff(f, x2, x2)))

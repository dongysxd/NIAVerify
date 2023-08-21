import sys
import gurobipy as gp
from gurobipy import GRB

def test1():
    model = gp.Model()
    vars = []
    for j in range ( 5 ):
        vars.append ( model . addVar (lb=0, ub=1))
    A = [1,1,1,1,1]
    expr = gp.LinExpr()
    for j in range(5):
        expr += A[j] * vars[j]
    #GRB. GREATER_EQUAL
    model.addLConstr(expr , GRB.LESS_EQUAL, -0.01)
    model.optimize()
    solution = []
    if model.status == GRB. OPTIMAL:
        x = model.getAttr ('X', vars )
        for i in range ( 5 ):
            solution.append (x[i])
        #print(solution)
        return True
    else :
        return False


print(test1())
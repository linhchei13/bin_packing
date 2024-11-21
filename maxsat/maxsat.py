from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

wcnf = WCNF()
wcnf.append([-1, -2])
wcnf.append([1], weight=4)
wcnf.append([2], weight=3)
with RC2(wcnf) as rc2:
    print(rc2.compute())
    print(rc2.cost)

def logical_or(x1, x2):
    return int(x1 or x2)

def logical_and(x1, x2):
    return int(x1 and x2)

def xor(x1, x2):
    return logical_or(x1, x2) and not logical_and(x1, x2)

#
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in inputs:
    print(f'XOR({x1}, {x2}) = {xor(x1, x2)}')

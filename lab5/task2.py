import numpy as np

def activation(x):
    return 1 if x >= 0 else 0

# AND
def compute_and(x1, x2):
    params = neurons['and']
    result = np.dot(params['weights'], [x1, x2]) - params['bias']
    return activation(result)

# OR
def compute_or(x1, x2):
    params = neurons['or']
    result = np.dot(params['weights'], [x1, x2]) - params['bias']
    return activation(result)

# XOR
def compute_xor(x1, x2):
    output_and = compute_and(x1, x2)
    output_or = compute_or(x1, x2)
    xor_params = neurons['xor']
    result = np.dot(xor_params['weights'], [output_or, output_and]) - xor_params['bias']
    return activation(result)

neurons = {
    'and': {'weights': np.array([1, 1]), 'bias': 1.5},
    'or': {'weights': np.array([1, 1]), 'bias': 0.5},
    'xor': {'weights': np.array([1, -1]), 'bias': 0.3}
}

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in inputs:
    print(f'XOR({x1}, {x2}) = {compute_xor(x1, x2)}')

def equation(neuron_type):
    params = neurons[neuron_type]
    w1, w2 = params['weights']
    bias = params['bias']
    if w2 != 0:
        print(f"Рівняння прямої для '{neuron_type}': {w1} * x1 + {w2} * x2 = {bias}")
    else:
        print(f"Рівняння прямої для '{neuron_type}': x1 = {bias / w1}")

equation('and')
equation('or')
equation('xor')

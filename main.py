import numpy as np
import matplotlib.pyplot as plt


class OptimizationMethods:

    def __init__(self, precision=10e-6, max_iters=1000, learning_rate=0.01):
        self.precision = precision
        self.max_iters = max_iters
        self.learning_rate = learning_rate

    def grad_descent(self, x, y, step):
        gradient = calculate_gradient(x, y, step)
        iterations = 0
        while self.max_iters > iterations or np.sqrt(gradient[0] ** 2 + gradient[1] ** 2) > self.precision:
            x -= step * gradient[0]
            y -= step * gradient[1]
            gradient = calculate_gradient(x, y, step)
            iterations += 1
        return x, y, func(x, y)

    def momentum(self, x, y, step, conservation_factor):
        gradient = calculate_gradient(x, y, step)
        iterations = 0
        while self.max_iters > iterations or np.sqrt(gradient[0] ** 2 + gradient[1] ** 2) > self.precision:
            x -= conservation_factor * gradient[0] + (1 - conservation_factor) * gradient[0] * step
            y -= conservation_factor * gradient[1] + (1 - conservation_factor) * gradient[1] * step
            gradient = calculate_gradient(x, y, step)
            iterations += 1
        return x, y, func(x, y)

    def nesterov_momentum(self, x, y, step):
        gradient = calculate_gradient(x, y, step)

    def ada_grad(self, x, y, step):
        gradient = calculate_gradient(x, y, step)

    def rms_prop(self, x, y, step):
        gradient = calculate_gradient(x, y, step)

    def adam(self, x, y, step):
        gradient = calculate_gradient(x, y, step)

    def newton(self, x, y, step):
        gradient = calculate_gradient(x, y, step)

    def gauss_newton(self, x, y, step):
        gradient = calculate_gradient(x, y, step)


def get_error_array(x, y, step, methods):
    plt.show()


def func(x, y):
    # return x ** 3 + np.sin(y)
    return (x ** 3) + 2 * (y ** 2) - (3 * x) - 4 * y


def numerical_derivative_x(x, y, step):
    return (func(x + step, y) - func(x - step, y)) / (2 * step)


def numerical_derivative_y(x, y, step):
    return (func(x, y + step) - func(x, y - step)) / (2 * step)


def calculate_gradient(x, y, step):
    return [numerical_derivative_x(x, y, step), numerical_derivative_y(x, y, step)]
    # return [analytical_derivative_x(x), analytical_derivative_y(y)]


def numerical_derivative_xx(x, y, step):
    return (numerical_derivative_x(x + step, y, step) - numerical_derivative_x(x - step, y, step)) / (2 * step)


def numerical_derivative_xy(x, y, step):
    return (numerical_derivative_x(x, y + step, step) - numerical_derivative_x(x, y - step, step)) / (2 * step)


def numerical_derivative_yy(x, y, step):
    return (numerical_derivative_y(x, y + step, step) - numerical_derivative_y(x, y - step, step)) / (2 * step)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    methods = OptimizationMethods(precision=0.01)
    print(methods.grad_descent(-0.5, -1, 0.1))

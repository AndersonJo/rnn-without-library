import numpy as np
from queue import deque
import pylab as plt
from keras.layers import SimpleRNN


def create_sin() -> np.array:
    data = np.arange(-1000, 1000, 0.3)
    data = np.sin(data).reshape(-1, 1)
    return data


def make_timeseries(data: np.array, x_size: int, y_size: int):
    N = data.shape[0]
    t_size = x_size + y_size
    dq = deque(maxlen=t_size)
    output_x = []
    output_y = []
    for i in range(N):
        dq.append(data[i])
        if len(dq) == t_size:
            t = np.array(dq)
            output_x.append(t[:x_size])
            output_y.append(t[x_size:])

    output_x = np.array(output_x)
    output_y = np.array(output_y)
    return output_x, output_y


def split_train_test(data: np.array, y_size: float = 0.2):
    N = len(data)
    test_size = int(N * 0.2)
    train_size = N - test_size
    train, test = data[:train_size], data[train_size:]
    return train, test


def visualize_random_data(x: np.array, y: np.array):
    N = x.shape[0]
    fig, plots = plt.subplots(3, 3)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    plots = plots.reshape(-1)

    for i, p in enumerate(plots):
        idx = np.random.randint(0, N)
        sample_x, sample_y = x[idx], y[idx]
        plot_x = np.arange(len(sample_x))
        plot_y = np.arange(len(sample_x), len(sample_x) + len(sample_y))
        p.plot(plot_x, sample_x, label='x')
        p.plot(plot_y, sample_y, label='y')
        p.legend()


def batch_loader(x: np.array, y: np.array, batch_size: int = 32):
    N = x.shape[0]
    for i in range(0, N, batch_size):
        sample_x = x[i:i + batch_size]
        sample_y = y[i:i + batch_size]
        yield sample_x, sample_y


def shuffle(x: np.array, y: np.array):
    N = x.shape[0]
    indices = np.random.permutation(np.arange(N))
    x = x[indices]
    y = y[indices]
    return x, y

SimpleRNN
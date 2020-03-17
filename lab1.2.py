import numpy as np
import random
from math import sin
import matplotlib.pyplot as plt
import time


def frequency(n, w):
    return w - n * temp


def expectation(signal, N):
    return sum(signal) / N


def dispersion(signal, mx, N):
    return sum((signal - mx) ** 2) / N


def autocorrelation_function(x, mx, N):
    R = np.zeros(int(N/2) - 1)
    for T in range(int(N/2) - 1):
        for t in range(int(N/2) - 1):
            R[T] = ((x[t] - mx) * (x[t + T] - mx))
    return np.sum(R)/(N-1)


def cross_correlation_function(x, y, N):
    R = np.zeros(int(N/2) - 1)
    for T in range(int(N/2) - 1):
        for t in range(int(N/2) - 1):
            R[T] = ((x[t] - expectation(x, N)) * (y[t + T] - expectation(y, N)))
    return np.sum(R)/(N-1)


n = 10
w = 1500
N = 256
temp = w / (n - 1)

# frequency
w_val = [frequency(n, w) for n in range(n)]
harmonics = np.empty(N)

# Harmonics generating
for n in range(n):
    A = random.randint(-10, 10)
    phi = random.randint(-360, 360)
    for t in range(N):
        harmonics[t] += A * sin(w_val[n] * t + phi)

mx = expectation(harmonics, N)
start = time.perf_counter()
autocor = autocorrelation_function(harmonics, mx, N)

time_rxx = time.perf_counter() - start

# Signal Y
y = np.empty(N)
for n in range(n):
    A = random.randint(-10, 10)
    phi = random.randint(-360, 360)
    for t in range(N):
        y[t] += A * sin(w_val[n] * t + phi)

Rxy = np.zeros(int(N/2) - 1)
for T in range(int(N/2) - 1):
    for t in range(int(N/2) - 1):
        Rxy[T] = ((harmonics[t] - mx) * (y[t + T] - expectation(y, N))) / (N - 1)
plt.figure(figsize=(10, 5))
plt.plot(Rxy, 'r')
plt.grid(True)
plt.show()

start = time.perf_counter()
cov = Rxy(harmonics, y, N)

time_rxy = time.perf_counter() - start

plt.figure(figsize=(30, 20))
plt.plot(harmonics, 'g')
plt.grid(True)
plt.show()

plt.figure(figsize=(30, 20))
plt.plot(y)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(Rxy, 'r')
plt.grid(True)
plt.show()

print(f"Rxx = {str(autocor)}")
print(f"Rxy = {str(cov)}")
print(f"Time Rxx = {str(time_rxx)}")
print(f"Time Rxy = {str(time_rxy)}")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def Euler(F, y0, t0, tN, h=0.001):
    n = int((tN - t0) / h)
    t = np.zeros((n + 1, 1))
    y = np.zeros((n + 1, len(y0)))
    t[0] = t0
    y[0, :] = y0

    for i in range(0, n):
        t[i + 1] = t[i] + h
        y[i + 1, :] = y[i, :] + h * np.array(F(t[i], y[i, :]))

    return t, y

def Trapecio(F, y0, t0, tN, h=0.001):
    n = int((tN - t0) / h)
    t = np.zeros((n + 1, 1))
    y = np.zeros((n + 1, len(y0)))
    t[0] = t0
    y[0, :] = y0

    for i in range(0, n):
        t[i + 1] = t[i] + h
        F1 = np.array(F(t[i], y[i, :]))
        F2 = np.array(F(t[i + 1], y[i, :] + h * F1))
        y[i + 1, :] = y[i, :] + 0.5 * h * (F1 + F2)

    return t, y

def PuntoMedio(F, y0, t0, tN, h=0.001):
    n = int((tN - t0) / h)
    t = np.zeros((n + 1, 1))
    y = np.zeros((n + 1, len(y0)))
    t[0] = t0
    y[0, :] = y0

    for i in range(0, n):
        t[i + 1] = t[i] + h
        t12 = t[i] + 0.5 * h
        y12 = y[i, :] + 0.5 * h * np.array(F(t[i], y[i, :]))
        y[i + 1, :] = y[i, :] + h * np.array(F(t12, y12))

    return t, y

def RK4(F, y0, t0, tN, h=0.001):
    n = int((tN - t0) / h)
    t = np.zeros((n + 1, 1))
    y = np.zeros((n + 1, len(y0)))
    t[0] = t0
    y[0, :] = y0

    for i in range(0, n):
        t[i + 1] = t[i] + h
        F1 = np.array(F(t[i], y[i, :]))
        F2 = np.array(F(t[i] + 0.5 * h, y[i, :] + 0.5 * h * F1))
        F3 = np.array(F(t[i] + 0.5 * h, y[i, :] + 0.5 * h * F2))
        F4 = np.array(F(t[i] + h, y[i, :] + h * F3))
        y[i + 1, :] = y[i, :] + (h / 6) * (F1 + 2 * F2 + 2 * F3 + F4)

    return t, y
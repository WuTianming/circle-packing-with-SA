# Simulated Annealing approach to solve the Circle Packing problem
# PB20000196 吴天铭

import sys

from math import log, sqrt, inf, exp, pi, cos, sin
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


################################### begin Params
zeroP_debug = ('p' in sys.argv)

pausetime = 0.001

if len(sys.argv) >= 2:
    n = int(sys.argv[1])
else:
    print("Please give `n' as a parameter.")
    sys.exit(1)

Tbegin = 0.3
Tend   = 0.0003
alpha  = 0.99

restart_ksi = 0.4

def Step(T):
    ksi = log(T/Tbegin, Tend/Tbegin)
    return Step.size / (1 + exp(Step.curvature * (ksi - Step.centerksi)))
Step.size      = 1.0
Step.curvature = 8.0
Step.centerksi = 0.45

def P(e, e1, T):
    if zeroP_debug: return 0
    return exp((e-e1) / T)
################################### end   Params


Magic = np.array([
    0.10,
    0.500000000000000000000000000000,
    0.292893218813452475599155637896,
    0.254333095030249817754744760429,
    0.250000000000000000000000000000,
    0.207106781186547524400844362105,
    0.187680601147476864319898426192,
    0.174457630187009438959427204500,
    0.170540688701054438818560595676,
    0.166666666666666666666666666667,
    0.148204322565228798668007362743,
    0.142399237695800384587114500527,
    0.139958844038428028961026945453,
    0.133993513499008491414263236065,
    0.129331793710034021408259201773,
    0.127166547515124908877372380214,
    0.125000000000000000000000000000,
])
Magic_diameter = 0.50 / Magic


def Find_diameter(points):
    span = points.max(axis=0) - points.min(axis=0)
    return max(span[0], span[1]) / 2 + 1.00

def Visualize(ax, points):
    ax.clear()
    rt, lb = points.max(axis=0), points.min(axis=0)
    span   = rt - lb
    center = (rt + lb) / 2
    diam   = max(span[0], span[1]) / 2 + 1.00
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-diam, diam)
    ax.set_ylim(-diam, diam)
    for p in points:
        ax.add_artist(Circle(xy=(p-center), radius=1, ec='b', fc=(0, 0, 0.7, 0.5)))
    plt.show()

def Relax(points, T):
    F = np.empty_like(points)
    F.fill(0)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i <= j:
                continue
            dr = p - q
            dist = sqrt(dr[0]**2 + dr[1]**2)
            # an additional force of T/10 is applied to ensure that circles get detached
            intensity = 2.00 - dist + T / 10.00
            if dist == 0.00:
                theta = random.random() * 2 * pi
                dF = intensity * np.array([cos(theta), sin(theta)])
                F[i] += dF
                F[j] -= dF
            elif dist < 2.00:
                dF = intensity * dr / dist
                F[i] += dF
                F[j] -= dF
    points += F / 5                  # move the circles
    # then calculate overlap value
    overlap = 0.00
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            dr = p - q
            dist = sqrt(dr[0]**2 + dr[1]**2)
            if (dist <= 2.00):
                overlap += (2.00 - dist)
    return overlap

def wander_one(points, T, pInd):
    step = Step(T)
    p = points[pInd]
    t = random.random() * pi * 2.00
    p += np.array((cos(t), sin(t))) * step
    while Relax(points, T) > 0.00:
        pass

def wander_all(points, T):
    step = Step(T) / 2
    for p in points:
        t = random.random() * pi * 2.00
        p += np.array((cos(t), sin(t))) * step * random.random()
    while Relax(points, T) > 0.00:
        pass

ax = axaccr = ""

def SA(points, Tb=Tbegin):
    T = Tb
    SA.best = min(SA.best, Find_diameter(points))
    inc, acc, rej = 0, 0, 0
    itr = 0
    idx = [i for i in range(0, n)]
    while T > Tend:
        itr += 1
        ksi = log(T/Tbegin, Tend/Tbegin)
        random.shuffle(idx)
        if itr % 25 == 0:
            accrate = 1.00 * (inc + acc) / (inc + acc + rej)
            print("[{:5.1%}] accRate = {:5.1%}".format(ksi, accrate))
            axaccr.plot(ksi, accrate, "gx")
            Visualize(ax, points)
            plt.pause(pausetime)
            inc = acc = rej = 0
        for i in range(0, n+1):
            prediam = min(SA.best * 1.5, Find_diameter(points))
            newpts = np.copy(points)
            if i != n:
                wander_one(newpts, T, idx[i])
            else:
                wander_all(newpts, T)
            newdiam = Find_diameter(newpts)
            if newdiam <= prediam:
                inc += 1
                SA.best = newdiam
                points[:] = newpts[:]
            else:
                if P(prediam, newdiam, T) > random.random():
                    acc += 1
                    points[:] = newpts[:]
                else:
                    rej += 1
        T *= alpha
    print("---------------------")
    print(" iterations = {}".format(itr))
    print(" optimal    = {}".format(SA.best))
    if Magic_diameter.size > n:
        print(" rel. err   = {:.8%}".format(SA.best / Magic_diameter[n] - 1.00))
    print("---------------------")
    Visualize(ax, points)
    plt.pause(0.01)

SA.best = inf

def main():
    global ax
    global axaccr

    random.seed()
    plt.ion()
    axes = plt.subplots(nrows=3)[1]
    ax = axes[0]
    axstep = axes[1]
    axaccr = axes[2]

    axstep.set_title("step - iteration")
    ISeq = np.arange(0, log(Tend/Tbegin, alpha), 5)
    TSeq = Tbegin * np.power(alpha, ISeq)
    StepSeq = np.array([Step(T) for T in TSeq])
    axstep.plot(ISeq, StepSeq)

    axaccr.set_title("Accept Rate - ksi")
    axaccr.set_xlim(0.0, 1.0)
    axaccr.set_ylim(0.0, 1.0)
    axaccr.format_coord = lambda x, y: 'x={:.2f}, y={:.2f}, T={:g}'.format(x, y, Tbegin * ((Tend/Tbegin) ** x))
    PSeq = np.array([P(0, 0.01, T) for T in TSeq])  # practical delta \approx 0.01
    axaccr.plot(ISeq / ISeq.max(), PSeq)

    plt.tight_layout()
    plt.show()

    pts = np.ndarray((n, 2))
    pts.fill(0)
    print("init {} points.".format(n))
    while Relax(pts, 1) > 0.00: pass

    Visualize(ax, pts)
    SA(pts)
    print(pts)
    input("Press Enter for next SA round...")
    while True:
        SA(pts, Tbegin * ((Tend / Tbegin) ** restart_ksi))
        print(pts)
        input("Press Enter for next SA round...")

if __name__ == '__main__':
    main()

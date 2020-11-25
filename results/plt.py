y = [0.5562, 0.0828, 0.0993, 0.2053, 0.0149, 0.0085, 0.0137, 0.0193]

import numpy as np
import os
import matplotlib.pyplot as plt

dir_path = './figs/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


plt.style.use('seaborn-bright')
N = 5
ind = np.arange(len(y))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, y, width)

plt.ylabel('Weight', fontsize=15)
plt.xlabel('Meta paths', fontsize=15)
plt.xticks(ind, ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'))
plt.tight_layout()
plt.savefig(dir_path + './weights.png', format='png', dpi=500)


l = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1', '5', '10']
y = [np.loadtxt('alpha_%s.txt' %s)[1][1] for s in l]
x = range(len(y))

def pltline(param):
    fig = plt.figure(figsize=(10, 5))
    # fig, ax = plt.subplots()
    marker = ['D', 's', 'O']
    markercolor = ['red', 'green']
    y_labels = param['y_labels']
    p = [121, 122]
    for i, y in enumerate(param['ys']):
        plt.subplot(p[i])
        plt.plot(range(len(y)), y, 'k', range(len(y)), y, marker[i], markerfacecolor=markercolor[i], markeredgecolor='black')
        plt.xticks(range(len(y)), param['range'])
        plt.ylabel(y_labels[i], fontsize=15)
        plt.xlabel(param['x_label'], fontsize=15)
    plt.tight_layout()
    plt.savefig(dir_path + param['name'] + '.png', format='png', dpi=500)

alpha_l = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1', '5', '10']
alpha = {'range' : alpha_l, 'ys' : [[np.loadtxt('alpha_%s.txt' %s)[1][1] for s in alpha_l], [np.loadtxt('alpha_%s.txt' %s)[1][2] for s in alpha_l]],
         'marker' : 'D', 'marker_color' : 'red', 'y_labels': [r'P@5', r'MRR@5'], 'x_label' : r'$\alpha$', 'name':'alpha'}

beta_l = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1', '5', '10']
beta = {'range' : beta_l, 'ys' : [[np.loadtxt('beta_%s.txt' %s)[1][1] for s in beta_l], [np.loadtxt('beta_%s.txt' %s)[1][2] for s in beta_l]],
         'marker' : 'D', 'marker_color' : 'red', 'y_labels': [r'P@5', r'MRR@5'], 'x_label' : r'$\beta$', 'name':'beta'}

negative_l = ['0.%d'%i for i in range(10)]
negative = {'range' : negative_l, 'ys' : [[np.loadtxt('negative_ratio_%s.txt' %s)[1][1] for s in negative_l],
                                          [np.loadtxt('negative_ratio_%s.txt' %s)[1][2] for s in negative_l]],
         'marker' : 'D', 'marker_color' : 'red', 'y_labels': [r'P@5', r'MRR@5'], 'x_label' : r'$\lambda$', 'name':'lambda'}

embed_l = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512]
embed = {'range' : embed_l, 'ys' : [[np.loadtxt('embed_%s.txt' %s)[1][1] for s in embed_l],
                                    [np.loadtxt('embed_%s.txt' %s)[1][2] for s in embed_l]],
         'marker' : 'D', 'marker_color' : 'red', 'y_labels': [r'P@5', r'MRR@5'], 'x_label' : r'Embedding size', 'name':'embedding'}

pltline(alpha)
pltline(beta)
pltline(negative)
pltline(embed)
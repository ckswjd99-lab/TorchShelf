import matplotlib.pyplot as plt

'''
[Log Example]

Epoch   1/100 | T LOSS: 1.6411, T ACC: 39.93%, V LOSS: 1.3920, V ACC: 49.55% | *
Epoch   2/100 | T LOSS: 1.2777, T ACC: 53.98%, V LOSS: 1.8302, V ACC: 44.26% |
Epoch   3/100 | T LOSS: 1.1174, T ACC: 60.15%, V LOSS: 1.1832, V ACC: 58.53% | *
Epoch   4/100 | T LOSS: 1.0317, T ACC: 63.08%, V LOSS: 1.1126, V ACC: 61.88% | *
Epoch   5/100 | T LOSS: 0.9669, T ACC: 65.63%, V LOSS: 1.2710, V ACC: 59.80% |
Epoch   6/100 | T LOSS: 0.9196, T ACC: 67.50%, V LOSS: 1.1341, V ACC: 63.06% | *
Epoch   7/100 | T LOSS: 0.8700, T ACC: 69.29%, V LOSS: 1.0461, V ACC: 66.35% | *
Epoch   8/100 | T LOSS: 0.8328, T ACC: 70.80%, V LOSS: 0.8913, V ACC: 69.57% | *
Epoch   9/100 | T LOSS: 0.7898, T ACC: 72.17%, V LOSS: 1.2997, V ACC: 60.09% |
Epoch  10/100 | T LOSS: 0.7652, T ACC: 73.35%, V LOSS: 0.9275, V ACC: 69.25% |
Epoch  11/100 | T LOSS: 0.7372, T ACC: 74.16%, V LOSS: 0.7719, V ACC: 73.20% | *
Epoch  12/100 | T LOSS: 0.7168, T ACC: 74.98%, V LOSS: 0.7915, V ACC: 73.01% |
Epoch  13/100 | T LOSS: 0.7031, T ACC: 75.64%, V LOSS: 0.8744, V ACC: 69.59% |
Epoch  14/100 | T LOSS: 0.6891, T ACC: 76.09%, V LOSS: 0.8561, V ACC: 71.61% |
Epoch  15/100 | T LOSS: 0.6716, T ACC: 76.59%, V LOSS: 0.7564, V ACC: 74.50% | *
Epoch  16/100 | T LOSS: 0.6628, T ACC: 77.05%, V LOSS: 0.9905, V ACC: 68.65% |
Epoch  17/100 | T LOSS: 0.6528, T ACC: 77.30%, V LOSS: 0.7253, V ACC: 75.37% | *
Epoch  18/100 | T LOSS: 0.6498, T ACC: 77.46%, V LOSS: 0.7574, V ACC: 73.97% |
Epoch  19/100 | T LOSS: 0.6422, T ACC: 77.82%, V LOSS: 0.9001, V ACC: 70.65% |
Epoch  20/100 | T LOSS: 0.6338, T ACC: 78.00%, V LOSS: 0.7971, V ACC: 73.93% |
Epoch  21/100 | T LOSS: 0.6276, T ACC: 78.30%, V LOSS: 0.7050, V ACC: 75.84% | *
Epoch  22/100 | T LOSS: 0.6177, T ACC: 78.64%, V LOSS: 0.8119, V ACC: 74.29% |
Epoch  23/100 | T LOSS: 0.6187, T ACC: 78.65%, V LOSS: 0.8395, V ACC: 72.00% |
Epoch  24/100 | T LOSS: 0.6104, T ACC: 79.02%, V LOSS: 0.6904, V ACC: 76.78% | *
Epoch  25/100 | T LOSS: 0.6087, T ACC: 78.77%, V LOSS: 0.9373, V ACC: 70.98% |
Epoch  26/100 | T LOSS: 0.6076, T ACC: 79.07%, V LOSS: 0.8266, V ACC: 74.67% |
Epoch  27/100 | T LOSS: 0.6038, T ACC: 79.10%, V LOSS: 0.6696, V ACC: 77.63% | *
...
'''

baseline_raw_path = './logs/train_pd_prate0.90_multistep.txt'
nprune_raw_path = './logs/train_npd_rg_prate0.90_multistep_02.txt'

baseline_raw = open(baseline_raw_path, 'r').readlines()
nprune_raw = open(nprune_raw_path, 'r').readlines()

baseline_vloss = []
baseline_vacc = []

nprune_vloss = []
nprune_vacc = []

baseline_best_vacc = 0
baseline_best_epoch = 0

for line in baseline_raw:
    if 'V LOSS' in line:
        vloss = float(line.split('V LOSS: ')[1].split(', V ACC: ')[0])
        vacc = float(line.split(', V ACC: ')[1].split('%')[0])
        baseline_vloss.append(vloss)
        baseline_vacc.append(vacc)

        if vacc > baseline_best_vacc:
            baseline_best_vacc = vacc
            baseline_best_epoch = len(baseline_vacc) - 1

nprune_best_vacc = 0
nprune_best_epoch = 0

for line in nprune_raw:
    if 'V LOSS' in line:
        vloss = float(line.split('V LOSS: ')[1].split(', V ACC: ')[0])
        vacc = float(line.split(', V ACC: ')[1].split('%')[0])
        nprune_vloss.append(vloss)
        nprune_vacc.append(vacc)

        if vacc > nprune_best_vacc:
            nprune_best_vacc = vacc
            nprune_best_epoch = len(nprune_vacc) - 1

# Plot the validation loss
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['figure.dpi'] = 400

plt.plot(baseline_vloss, label='PD-Prune')
plt.plot(nprune_vloss, label='NPD-Prune')
plt.title('Validation Loss')

plt.scatter(baseline_best_epoch, baseline_vloss[baseline_best_epoch], color='red')
plt.annotate(f'{baseline_best_vacc:.2f}%', (baseline_best_epoch, baseline_vloss[baseline_best_epoch]), textcoords="offset points", xytext=(-20,10), ha='center', color='C0')

plt.scatter(nprune_best_epoch, nprune_vloss[nprune_best_epoch], color='red')
plt.annotate(f'{nprune_best_vacc:.2f}%', (nprune_best_epoch, nprune_vloss[nprune_best_epoch]), textcoords="offset points", xytext=(20,10), ha='center', color='C1')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0.4, 2)
plt.legend()

plt.savefig('./logs/pd_vs_npd_vloss.png')
plt.close()


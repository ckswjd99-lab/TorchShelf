import matplotlib.pyplot as plt

'''
Examples

Baseline
Epoch   1/100 | T LOSS: 1.7936, T ACC: 35.99%, V LOSS: 1.0747, V ACC: 61.71% | *                              
Epoch   2/100 | T LOSS: 1.1900, T ACC: 57.87%, V LOSS: 0.7497, V ACC: 73.24% | *                              
Epoch   3/100 | T LOSS: 1.0104, T ACC: 64.63%, V LOSS: 0.7206, V ACC: 74.77% | *                              
Epoch   4/100 | T LOSS: 0.8849, T ACC: 68.98%, V LOSS: 0.6395, V ACC: 77.82% | *                              
Epoch   5/100 | T LOSS: 0.8077, T ACC: 72.01%, V LOSS: 0.5406, V ACC: 81.55% | *                              
Epoch   6/100 | T LOSS: 0.7476, T ACC: 73.92%, V LOSS: 0.4948, V ACC: 83.06% | *                              
Epoch   7/100 | T LOSS: 0.7042, T ACC: 75.65%, V LOSS: 0.4724, V ACC: 83.43% | *                              
Epoch   8/100 | T LOSS: 0.6618, T ACC: 77.05%, V LOSS: 0.4488, V ACC: 84.44% | *                              
Epoch   9/100 | T LOSS: 0.6190, T ACC: 78.47%, V LOSS: 0.4358, V ACC: 84.94% | *                              
Epoch  10/100 | T LOSS: 0.5930, T ACC: 79.45%, V LOSS: 0.4041, V ACC: 86.16% | *                              
Epoch  11/100 | T LOSS: 0.5672, T ACC: 80.33%, V LOSS: 0.4016, V ACC: 86.26% | *                              
Epoch  12/100 | T LOSS: 0.5431, T ACC: 81.24%, V LOSS: 0.3786, V ACC: 87.29% | *                              
Epoch  13/100 | T LOSS: 0.5186, T ACC: 81.87%, V LOSS: 0.3952, V ACC: 87.37% | *                              
Epoch  14/100 | T LOSS: 0.4969, T ACC: 82.75%, V LOSS: 0.3571, V ACC: 88.12% | *                              
Epoch  15/100 | T LOSS: 0.4788, T ACC: 83.41%, V LOSS: 0.3519, V ACC: 88.38% | *                              
Epoch  16/100 | T LOSS: 0.4677, T ACC: 83.63%, V LOSS: 0.3273, V ACC: 89.43% | *                              
Epoch  17/100 | T LOSS: 0.4478, T ACC: 84.49%, V LOSS: 0.3196, V ACC: 89.57% | *                              
Epoch  18/100 | T LOSS: 0.4302, T ACC: 84.92%, V LOSS: 0.3051, V ACC: 89.91% | *                              
Epoch  19/100 | T LOSS: 0.4286, T ACC: 85.14%, V LOSS: 0.3061, V ACC: 89.61% |                                
Epoch  20/100 | T LOSS: 0.4155, T ACC: 85.60%, V LOSS: 0.3212, V ACC: 89.71% |       

Grow
Epoch   1/100 | T LOSS: 1.8833, T ACC: 32.07%, V LOSS: 1.4122, V ACC: 49.63% | PARAM     27,140 (1.1087%)     
Epoch   2/100 | T LOSS: 1.6144, T ACC: 42.00%, V LOSS: 1.1832, V ACC: 57.98% | PARAM     51,446 (2.1016%)     
Epoch   3/100 | T LOSS: 1.4559, T ACC: 48.06%, V LOSS: 1.0523, V ACC: 62.99% | PARAM     75,713 (3.0929%)     
Epoch   4/100 | T LOSS: 1.3301, T ACC: 52.83%, V LOSS: 0.9196, V ACC: 67.22% | PARAM     99,945 (4.0828%)     
Epoch   5/100 | T LOSS: 1.2243, T ACC: 56.78%, V LOSS: 0.8470, V ACC: 70.53% | PARAM    124,177 (5.0727%)     
Epoch   6/100 | T LOSS: 1.1411, T ACC: 59.73%, V LOSS: 0.7767, V ACC: 72.88% | PARAM    148,396 (6.0621%)     
Epoch   7/100 | T LOSS: 1.0651, T ACC: 62.42%, V LOSS: 0.7289, V ACC: 74.77% | PARAM    172,608 (7.0511%)     
Epoch   8/100 | T LOSS: 1.0056, T ACC: 64.59%, V LOSS: 0.6749, V ACC: 76.51% | PARAM    196,822 (8.0403%)     
Epoch   9/100 | T LOSS: 0.9450, T ACC: 66.87%, V LOSS: 0.6373, V ACC: 77.99% | PARAM    221,030 (9.0292%)     
Epoch  10/100 | T LOSS: 0.9195, T ACC: 67.76%, V LOSS: 0.6147, V ACC: 78.77% | PARAM    245,242 (10.0183%)    
Epoch  11/100 | T LOSS: 0.8786, T ACC: 69.30%, V LOSS: 0.5865, V ACC: 79.78% | PARAM    269,448 (11.0071%)    
Epoch  12/100 | T LOSS: 0.8483, T ACC: 70.35%, V LOSS: 0.5814, V ACC: 80.05% | PARAM    293,656 (11.9960%)    
Epoch  13/100 | T LOSS: 0.8139, T ACC: 71.49%, V LOSS: 0.5688, V ACC: 80.85% | PARAM    317,864 (12.9849%)    
Epoch  14/100 | T LOSS: 0.7934, T ACC: 72.18%, V LOSS: 0.5313, V ACC: 81.86% | PARAM    342,072 (13.9738%)    
Epoch  15/100 | T LOSS: 0.7639, T ACC: 73.38%, V LOSS: 0.5194, V ACC: 82.48% | PARAM    366,278 (14.9627%)    
Epoch  16/100 | T LOSS: 0.7463, T ACC: 74.04%, V LOSS: 0.5127, V ACC: 82.58% | PARAM    390,485 (15.9515%)    
Epoch  17/100 | T LOSS: 0.7266, T ACC: 74.51%, V LOSS: 0.4852, V ACC: 83.30% | PARAM    414,690 (16.9403%)    
Epoch  18/100 | T LOSS: 0.7102, T ACC: 75.27%, V LOSS: 0.4760, V ACC: 83.82% | PARAM    438,897 (17.9292%)    
Epoch  19/100 | T LOSS: 0.6934, T ACC: 75.77%, V LOSS: 0.4612, V ACC: 84.40% | PARAM    463,101 (18.9179%)    
Epoch  20/100 | T LOSS: 0.6721, T ACC: 76.34%, V LOSS: 0.4496, V ACC: 84.71% | PARAM    487,307 (19.9068%)    
Epoch  21/100 | T LOSS: 0.6545, T ACC: 77.20%, V LOSS: 0.4592, V ACC: 84.32% | PARAM    511,511 (20.8955%)    
Epoch  22/100 | T LOSS: 0.6436, T ACC: 77.49%, V LOSS: 0.4534, V ACC: 84.68% | PARAM    535,719 (21.8844%)    
Epoch  23/100 | T LOSS: 0.6236, T ACC: 78.35%, V LOSS: 0.4209, V ACC: 85.60% | PARAM    559,923 (22.8732%)    
Epoch  24/100 | T LOSS: 0.6096, T ACC: 78.76%, V LOSS: 0.4387, V ACC: 85.33% | PARAM    584,128 (23.8620%)    
Epoch  25/100 | T LOSS: 0.5973, T ACC: 79.09%, V LOSS: 0.4186, V ACC: 85.71% | PARAM    608,332 (24.8507%)    
Epoch  26/100 | T LOSS: 0.5902, T ACC: 79.47%, V LOSS: 0.4093, V ACC: 85.70% | PARAM    632,541 (25.8397%)    
Epoch  27/100 | T LOSS: 0.5761, T ACC: 79.90%, V LOSS: 0.3957, V ACC: 86.54% | PARAM    656,745 (26.8284%)    
Epoch  28/100 | T LOSS: 0.5629, T ACC: 80.44%, V LOSS: 0.3771, V ACC: 87.02% | PARAM    680,951 (27.8172%)    
Epoch  29/100 | T LOSS: 0.5594, T ACC: 80.64%, V LOSS: 0.3877, V ACC: 86.83% | PARAM    705,154 (28.8059%)    
Epoch  30/100 | T LOSS: 0.5441, T ACC: 80.96%, V LOSS: 0.3968, V ACC: 86.54% | PARAM    729,361 (29.7948%)    
Epoch  31/100 | T LOSS: 0.5320, T ACC: 81.41%, V LOSS: 0.3789, V ACC: 87.23% | PARAM    753,566 (30.7836%)    
Epoch  32/100 | T LOSS: 0.5261, T ACC: 81.67%, V LOSS: 0.3794, V ACC: 86.98% | PARAM    777,772 (31.7724%)    
Epoch  33/100 | T LOSS: 0.5179, T ACC: 81.92%, V LOSS: 0.3578, V ACC: 87.98% | PARAM    801,976 (32.7612%)    
Epoch  34/100 | T LOSS: 0.5083, T ACC: 82.00%, V LOSS: 0.3807, V ACC: 87.50% | PARAM    826,183 (33.7501%)    
Epoch  35/100 | T LOSS: 0.4937, T ACC: 82.97%, V LOSS: 0.3453, V ACC: 88.29% | PARAM    850,387 (34.7388%)    
Epoch  36/100 | T LOSS: 0.4903, T ACC: 82.95%, V LOSS: 0.3570, V ACC: 88.28% | PARAM    874,593 (35.7276%)    
Epoch  37/100 | T LOSS: 0.4790, T ACC: 83.52%, V LOSS: 0.3353, V ACC: 88.69% | PARAM    898,801 (36.7165%)    
Epoch  38/100 | T LOSS: 0.4703, T ACC: 83.66%, V LOSS: 0.3429, V ACC: 88.61% | PARAM    923,005 (37.7053%)    
Epoch  39/100 | T LOSS: 0.4628, T ACC: 83.98%, V LOSS: 0.3348, V ACC: 88.95% | PARAM    947,211 (38.6941%)    
Epoch  40/100 | T LOSS: 0.4560, T ACC: 84.00%, V LOSS: 0.3435, V ACC: 88.57% | PARAM    971,414 (39.6828%)    
Epoch  41/100 | T LOSS: 0.4481, T ACC: 84.34%, V LOSS: 0.3385, V ACC: 88.95% | PARAM    995,622 (40.6717%)    
Epoch  42/100 | T LOSS: 0.4447, T ACC: 84.47%, V LOSS: 0.3188, V ACC: 89.17% | PARAM  1,019,827 (41.6605%)    
Epoch  43/100 | T LOSS: 0.4376, T ACC: 84.84%, V LOSS: 0.3224, V ACC: 89.48% | PARAM  1,044,032 (42.6493%)    
Epoch  44/100 | T LOSS: 0.4324, T ACC: 84.80%, V LOSS: 0.3224, V ACC: 89.38% | PARAM  1,068,237 (43.6381%)    
Epoch  45/100 | T LOSS: 0.4253, T ACC: 85.10%, V LOSS: 0.3260, V ACC: 88.96% | PARAM  1,092,443 (44.6269%)    
Epoch  46/100 | T LOSS: 0.4184, T ACC: 85.36%, V LOSS: 0.3123, V ACC: 89.66% | PARAM  1,116,647 (45.6157%)    
Epoch  47/100 | T LOSS: 0.4134, T ACC: 85.50%, V LOSS: 0.3209, V ACC: 89.97% | PARAM  1,140,853 (46.6045%)    
Epoch  48/100 | T LOSS: 0.4046, T ACC: 85.84%, V LOSS: 0.2917, V ACC: 90.23% | PARAM  1,165,058 (47.5933%)    
Epoch  49/100 | T LOSS: 0.4022, T ACC: 85.93%, V LOSS: 0.3053, V ACC: 89.90% | PARAM  1,189,265 (48.5822%)    
Epoch  50/100 | T LOSS: 0.3990, T ACC: 86.03%, V LOSS: 0.3080, V ACC: 90.03% | PARAM  1,213,470 (49.5709%)    

'''

baseline_log_path = './logs/train_fo.txt'
grow_log_path = './logs/train_grow_fo.txt'

baseline_log = open(baseline_log_path, 'r')
grow_log = open(grow_log_path, 'r')

baseline_log_lines = baseline_log.readlines()
grow_log_lines = grow_log.readlines()

baseline_log.close()
grow_log.close()

baseline_val_loss = []
baseline_val_acc = []
baseline_num_param = [2447946 * i for i in range(1, 101)]

baseline_best_acc = 0
baseline_best_epoch = 0

grow_val_loss = []
grow_val_acc = []
grow_num_param = [0]

grow_best_acc = 0
grow_best_epoch = 0


for line in baseline_log_lines:
    if 'V LOSS' in line:
        loss = float(line.split('V LOSS: ')[1].split(',')[0])
        acc = float(line.split('V ACC: ')[1].split('%')[0])
        
        baseline_val_loss.append(loss)
        baseline_val_acc.append(acc)
        
        if acc > baseline_best_acc:
            baseline_best_acc = acc
            baseline_best_epoch = int(line.split('Epoch ')[1].split('/')[0])


for line in grow_log_lines:
    if 'V LOSS' in line:
        loss = float(line.split('V LOSS: ')[1].split(',')[0])
        acc = float(line.split('V ACC: ')[1].split('%')[0])
        num_param = int(line.split('PARAM ')[1].split(' (')[0].replace(',', ''))

        grow_val_loss.append(loss)
        grow_val_acc.append(acc)
        grow_num_param.append(num_param + grow_num_param[-1])
        
        if acc > grow_best_acc:
            grow_best_acc = acc
            grow_best_epoch = int(line.split('Epoch ')[1].split('/')[0])

grow_num_param = grow_num_param[1:]

# plot graph
# x-axis: number of parameters
# y-axis: validation loss

plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['figure.dpi'] = 400

plt.figure()
plt.plot(baseline_num_param, baseline_val_loss, label='Baseline')
plt.plot(grow_num_param, grow_val_loss, label='Growing')

# draw red point for the best validation accuracy, then draw a arrow to the point, and write the accuracy
plt.plot(baseline_num_param[baseline_best_epoch-1], baseline_val_loss[baseline_best_epoch-1], 'ro')
plt.plot(grow_num_param[grow_best_epoch-1], grow_val_loss[grow_best_epoch-1], 'ro')
plt.annotate(f'{baseline_best_acc:.2f}%', (baseline_num_param[baseline_best_epoch-1], baseline_val_loss[baseline_best_epoch-1]), textcoords="offset points", xytext=(0,20), ha='center', color='C0')
plt.annotate(f'{grow_best_acc:.2f}%', (grow_num_param[grow_best_epoch-1], grow_val_loss[grow_best_epoch-1]), textcoords="offset points", xytext=(0,20), ha='center', color='C1')

plt.xlabel('Number of Parameter Updates')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Validation Loss')
plt.savefig('./val_loss.png')

# plot graph
# x-axis: size of model
# y-axis: validation accuracy

plt.figure()
plt.plot(grow_num_param, grow_val_acc, label='Growing')

plt.xlabel('Size of Model')
plt.ylabel('Validation Accuracy')
plt.savefig('./val_acc.png')

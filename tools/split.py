import sys

with open(sys.argv[1], 'r') as f, open(sys.argv[1] + '.pred', 'w') as f_pred, open(sys.argv[1] + '.true', 'w') as f_true:
    for line in f:
        if '\t' not in line:
            continue
        pred, true = line.split('\t')
        if not pred:
            pred = '<unk>'
        if not true.strip():
            true = '<unk>'
        f_pred.write(pred + '\n')
        f_true.write(true.strip() + '\n')


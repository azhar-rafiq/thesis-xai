"""
usage: python check_log.py [log_file]
       python check_log.py          # picks the most recent train_mult_rsna_*.log
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

log_dir = Path(__file__).parent

if len(sys.argv) > 1:
    log_path = Path(sys.argv[1])
else:
    logs = sorted(log_dir.glob('train_mult_rsna_*.log'))
    if not logs:
        print('No log files found.')
        sys.exit(1)
    log_path = logs[-1]

print(f'Log: {log_path.name}')
print(f'Size: {log_path.stat().st_size / 1e6:.1f} MB')
print()

content = log_path.read_bytes().decode('utf-8', errors='ignore')

# -- status -------------------------------------------------------------------
done   = 'Done!' in content
killed = 'Killed' in content or 'oom_kill' in content
oom    = 'OutOfMemory' in content or 'OOM' in content

if done:
    status = 'DONE'
elif killed and oom:
    status = 'OOM KILLED'
elif killed:
    status = 'KILLED'
else:
    status = 'RUNNING'

print(f'Status: {status}')

# -- timing -------------------------------------------------------------------
start_match = re.search(r'Job started: (.+)', content)
if start_match:
    try:
        started = datetime.strptime(start_match.group(1).strip(), '%a %d %b %H:%M:%S %Z %Y')
        elapsed = datetime.now() - started
        h, rem = divmod(int(elapsed.total_seconds()), 3600)
        m = rem // 60
        print(f'Started: {started.strftime("%Y-%m-%d %H:%M")}  |  Elapsed: {h}h {m}m')
    except ValueError:
        started = None
else:
    started = None

print()

# -- epochs and val_auc -------------------------------------------------------
epochs   = re.findall(r'Epoch (\d+)/(\d+)', content)
val_aucs = re.findall(r'val_auc: ([0-9.]+)', content)

current_epoch = int(epochs[-1][0]) if epochs else 0
total_epochs  = int(epochs[-1][1]) if epochs else 30

if epochs:
    print(f'Epoch: {current_epoch}/{total_epochs}')

if val_aucs:
    print(f'\nval_auc per completed epoch:')
    best = max(val_aucs, key=float)
    for i, v in enumerate(val_aucs, 1):
        marker = ' <-- best' if v == best else ''
        print(f'  epoch {i:>2}: {v}{marker}')
    print(f'\nBest val_auc so far: {best}')

# -- current step speed and epoch ETA ----------------------------------------
steps = re.findall(r'(\d+)/(\d+)[^\n]*?(\d+)ms/step', content)
epoch_min = None
if steps:
    last_step, total_steps, ms = steps[-1]
    ms_per_step   = int(ms)
    total_steps_n = int(total_steps)
    last_step_n   = int(last_step)
    remaining_steps = total_steps_n - last_step_n
    epoch_min = total_steps_n * ms_per_step / 1000 / 60
    eta_epoch_min = remaining_steps * ms_per_step / 1000 / 60
    print(f'\nSpeed: {ms}ms/step  |  ~{epoch_min:.0f} min/epoch')
    print(f'Current step: {last_step}/{total_steps}  (~{eta_epoch_min:.0f} min to this epoch end)')

# -- finish estimate ----------------------------------------------------------
if epoch_min and current_epoch > 0:
    # assume early stopping fires after at most `patience` more non-improving
    # epochs; conservatively estimate 5 more epochs from here
    patience = 5
    epochs_left = min(patience, total_epochs - current_epoch)
    finish_estimate = datetime.now() + timedelta(minutes=eta_epoch_min + epochs_left * epoch_min)
    print(f'\nEstimated finish: {finish_estimate.strftime("%Y-%m-%d %H:%M")}  '
          f'(~{eta_epoch_min + epochs_left * epoch_min:.0f} min remaining, assuming {epochs_left} more epochs)')

# -- checkpoints --------------------------------------------------------------
checkpoints = re.findall(r'(retrain_checkpoint[^\s\]]+\.keras)', content)
unique_checkpoints = list(dict.fromkeys(checkpoints))
if unique_checkpoints:
    saves = len(re.findall(r'val_auc improved', content))
    print(f'\nCheckpoints saved: {saves}')
    print(f'  latest: {unique_checkpoints[-1]}')

# -- final results (if done) --------------------------------------------------
if done:
    test_aucs = re.findall(r'([\w ]+?)\s{2,}AUC: ([0-9.]+)', content)
    if test_aucs:
        print('\nTest AUCs:')
        for label, auc in test_aucs:
            print(f'  {label.strip():<25} {auc}')
    saved = re.findall(r'Saved\s+(\S+)', content)
    if saved:
        print('\nSaved files:')
        for s in saved:
            print(f'  {s}')

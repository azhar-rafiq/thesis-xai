"""
usage: python check_log.py [log_file]
       python check_log.py          # picks the most recent train_*.log
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

log_dir = Path(__file__).parent

if len(sys.argv) > 1:
    log_path = Path(sys.argv[1])
else:
    logs = sorted(log_dir.glob('train_*.log'), key=lambda p: p.stat().st_mtime)
    if not logs:
        print('No log files found.')
        sys.exit(1)
    log_path = logs[-1]

print(f'Log: {log_path.name}')
print(f'Size: {log_path.stat().st_size / 1e6:.1f} MB')
print()

raw = log_path.read_bytes().decode('utf-8', errors='ignore')

# strip ANSI escape codes so all regex below work cleanly
content = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw)

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
started = None
if start_match:
    raw_ts = start_match.group(1).strip()
    # normalise double spaces (e.g. "Fri  5 Jun" for single-digit days)
    raw_ts = re.sub(r'  +', ' ', raw_ts)
    for fmt in ('%a %d %b %H:%M:%S %Z %Y', '%a %b %d %H:%M:%S %Z %Y'):
        try:
            started = datetime.strptime(raw_ts, fmt)
            break
        except ValueError:
            continue
    if started:
        elapsed = datetime.now() - started
        h, rem = divmod(int(elapsed.total_seconds()), 3600)
        m = rem // 60
        print(f'Started: {started.strftime("%Y-%m-%d %H:%M")}  |  Elapsed: {h}h {m}m')

print()

# -- grid search / params -----------------------------------------------------
skip_match = re.search(r'Skipping grid search.*?params: ({.+?})', content)
best_match  = re.search(r'Best params: ({.+?})', content)
cv_match    = re.search(r'Best CV score[^:]*: ([0-9.]+)', content)

gs_started  = 'Grid Search started' in content
gs_finished = 'Grid Search finished' in content
gs_running  = gs_started and not gs_finished

if skip_match:
    print(f'Grid search: skipped (hardcoded params)')
    try:
        import ast
        params = ast.literal_eval(skip_match.group(1))
        short = {k.split('__')[-1]: v for k, v in params.items()}
        print(f'  params: {short}')
    except Exception:
        pass
elif gs_running:
    fits_done = len(re.findall(r'\[CV\] END', content))
    print(f'Grid search: RUNNING  ({fits_done} CV fits done so far)')
elif best_match:
    print(f'Grid search: done')
    try:
        import ast
        params = ast.literal_eval(best_match.group(1))
        short = {k.split('__')[-1]: v for k, v in params.items()}
        print(f'  best params: {short}')
    except Exception:
        pass
    if cv_match:
        print(f'  best CV score: {cv_match.group(1)}')

print()

# -- epochs and val_auc -------------------------------------------------------
epochs   = re.findall(r'Epoch (\d+)/(\d+)', content)

current_epoch = int(epochs[-1][0]) if epochs else 0
total_epochs  = int(epochs[-1][1]) if epochs else 30

if epochs:
    print(f'Epoch: {current_epoch}/{total_epochs}')

# only show val_aucs from the retrain phase (after grid search finishes)
retrain_start = re.search(r'(Retraining best model|Starting final retrain|Retrain phase|Retrain started)', content)
all_val_aucs  = re.findall(r'val_auc: ([0-9.]+)', content)
if gs_running:
    val_aucs = []  # fold val_aucs would be misleading while grid search is in progress
elif retrain_start:
    auc_section = content[retrain_start.start():]
    val_aucs    = re.findall(r'val_auc: ([0-9.]+)', auc_section)
else:
    # no grid search marker -- take the last total_epochs entries (retrain at end)
    val_aucs = all_val_aucs[-total_epochs:] if len(all_val_aucs) > total_epochs else all_val_aucs

if val_aucs:
    print(f'\nval_auc per retrain epoch:')
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
    ms_per_step    = int(ms)
    total_steps_n  = int(total_steps)
    last_step_n    = int(last_step)
    remaining_steps = total_steps_n - last_step_n
    epoch_min      = total_steps_n * ms_per_step / 1000 / 60
    eta_epoch_min  = remaining_steps * ms_per_step / 1000 / 60
    print(f'\nSpeed: {ms}ms/step  |  ~{epoch_min:.0f} min/epoch')
    print(f'Current step: {last_step}/{total_steps}  (~{eta_epoch_min:.0f} min to this epoch end)')

# -- finish estimate ----------------------------------------------------------
if not done and epoch_min and current_epoch > 0:
    patience      = 5
    epochs_left   = min(patience, total_epochs - current_epoch)
    eta_total_min = eta_epoch_min + epochs_left * epoch_min
    finish_estimate = datetime.now() + timedelta(minutes=eta_total_min)
    print(f'\nEstimated finish: {finish_estimate.strftime("%Y-%m-%d %H:%M")}  '
          f'(~{eta_total_min:.0f} min remaining, assuming {epochs_left} more epochs)')

# -- checkpoints --------------------------------------------------------------
ckpt_pattern = re.findall(
    r'((?:rsna_cnn|rsna_vit|vit_checkpoint|retrain_checkpoint)[^\s\]]+\.keras)',
    content
)
unique_checkpoints = list(dict.fromkeys(ckpt_pattern))
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

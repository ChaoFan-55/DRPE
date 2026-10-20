"""Sequential unattended execution with failure gating, logs and idle-sleep guard.

Only this baseline's results directory is written. This does not schedule a
background task or change persistent Windows power settings.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


@contextmanager
def keep_awake(enabled):
    previous = None
    function = None
    if enabled and os.name == 'nt':
        import ctypes
        function = ctypes.WinDLL('kernel32', use_last_error=True).SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        previous = function(0x80000001)  # continuous + system required; screen may sleep
        if not previous:
            raise OSError(ctypes.get_last_error(), 'Windows idle-sleep prevention failed.')
    try:
        yield
    finally:
        if function is not None and previous:
            function(previous)


@contextmanager
def single_instance(path):
    """An OS file lock is released even if the runner is forcibly closed."""
    with Path(path).open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('An overnight run already holds this baseline lock. Do not start a second run.') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def stages(root, device):
    def command(script, *args):
        return [sys.executable, '-u', '-B', str(PACKAGE/script),
                '--project-root', str(root), *args]
    package_literal = repr(str(PACKAGE))
    environment_check = (
        f'import sys; sys.path.insert(0, {package_literal}); import common; '
        'import torch, torchvision, lpips; '
        'print("Python:", sys.executable); print("Torch:", torch.__version__); '
        'print("CUDA available:", torch.cuda.is_available()); '
        + ('assert torch.cuda.is_available(), "CUDA is unavailable"; ' if device == 'cuda' else '')
        + 'lpips.LPIPS(net="alex").eval(); print("Environment and LPIPS: PASS")'
    )
    yield '00_environment', [sys.executable, '-u', '-B', '-c', environment_check]
    yield '01_preflight', command('preflight.py', '--check-data')
    smoke_args = ['--seed', '42', '--device', device, '--smoke']
    if (PACKAGE/'results/smoke/seed42/resume.pt').is_file():
        smoke_args.append('--resume')
    yield '02_smoke_training', command('train.py', *smoke_args)
    for attack in ['dlg', 'idlg']:
        yield f'02_smoke_{attack}', command('attack.py', '--seed', '42', '--target', 'T00',
                   '--restart', '0', '--attack', attack, '--device', device, '--smoke', '--skip-lpips')
    for seed in [42, 43, 44]:
        train_args = ['--seed', str(seed), '--device', device]
        if (PACKAGE/f'results/formal/seed{seed}/resume.pt').is_file():
            train_args.append('--resume')
        yield f'03_training_seed{seed}', command('train.py', *train_args)
    yield '04_formal_attacks', command('run_attacks.py', '--device', device)
    yield '05_comparison', command('compare.py')


def save_status(path, status):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def validate_existing_config():
    current = json.loads((PACKAGE/'config.json').read_text(encoding='utf-8-sig'))
    for mode in ['smoke', 'formal']:
        for seed in [42, 43, 44]:
            signature = PACKAGE/f'results/{mode}/seed{seed}/run_signature.json'
            if signature.is_file():
                saved = json.loads(signature.read_text(encoding='utf-8-sig'))
                if saved.get('defense') != current:
                    raise ValueError(f'Defense configuration changed: {signature}. Do not mix experiment variants.')


def run_stage(name, command, env, log_path, status_path, status):
    status.update(status='running', stage=name, command=command,
                  stage_started=datetime.now().astimezone().isoformat())
    save_status(status_path, status)
    line = f'\n[{status["stage_started"]}] START {name}\n{subprocess.list2cmdline(command)}\n'
    with log_path.open('a', encoding='utf-8') as log:
        print(line, end='', flush=True)
        log.write(line); log.flush()
        process = subprocess.Popen(command, cwd=PACKAGE, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == 'nt':
                    # Stop only the subprocess tree launched by this stage.
                    # This also stops the attack child of run_attacks.py.
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
        log.write(f'[{datetime.now().astimezone().isoformat()}] END {name} exit={code}\n')
    if code != 0:
        raise RuntimeError(f'{name} exited with code {code}; subsequent stages were NOT started.')
    status['completed_stages'].append(name)
    save_status(status_path, status)


def main():
    parser = argparse.ArgumentParser(description='Run the baseline sequentially overnight; stop on errors.')
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-idle-sleep', action='store_true')
    args = parser.parse_args()
    root = args.project_root.resolve() if args.project_root else PACKAGE.parent
    if not (root/'src/models.py').is_file():
        raise FileNotFoundError('The parent must be drpe_formal_experiments, or specify --project-root.')
    for script in ['preflight.py', 'train.py', 'attack.py', 'run_attacks.py', 'compare.py', 'config.json']:
        if not (PACKAGE/script).is_file():
            raise FileNotFoundError(f'Missing baseline file: {script}')
    validate_existing_config()
    if args.dry_run:
        for name, command in stages(root, args.device):
            print(name, subprocess.list2cmdline(command))
        return 0
    folder = PACKAGE/'results/overnight'
    folder.mkdir(parents=True, exist_ok=True)
    token = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    status_path = folder/f'status_{token}.json'
    log_path = folder/f'console_{token}.log'
    status = dict(status='starting', started=datetime.now().astimezone().isoformat(), pid=os.getpid(),
                  python=sys.executable, project_root=str(root), log=str(log_path),
                  completed_stages=[])
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    print(f'Log: {log_path}\nStatus: {status_path}\nKeep this terminal and PyCharm open.', flush=True)
    try:
        with single_instance(folder/'active.lock'), keep_awake(not args.allow_idle_sleep):
            for name, command in stages(root, args.device):
                run_stage(name, command, env, log_path, status_path, status)
        status.update(status='complete', finished=datetime.now().astimezone().isoformat())
        save_status(status_path, status)
        print(f'ALL STAGES COMPLETE. Comparison: {PACKAGE/"results/comparison"}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=str(error), finished=datetime.now().astimezone().isoformat())
        save_status(status_path, status)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nSTOPPED: {type(error).__name__}: {error}\n')
        print(f'STOPPED at {status.get("stage")}: {error}\nRead {log_path}. Rerun the same command after fixing the cause.', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

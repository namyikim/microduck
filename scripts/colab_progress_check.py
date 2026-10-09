from colab_cli.common import state
from colab_cli.runtime import ColabRuntime
from colab_cli.commands.execution import display_output
import sys
import json
from pathlib import Path

if len(sys.argv) > 1:
    session_name = sys.argv[1]
else:
    monitor_path = Path('/private/tmp/microduck_monitor_state.json')
    try:
        session_name = json.loads(monitor_path.read_text())['session']
        if not isinstance(session_name, str) or not session_name.strip():
            raise ValueError('Invalid session name')
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit('최신 세션 정보를 읽을 수 없습니다. 스크립트 뒤에 세션명을 지정해 주세요.')
print('CHECKING_SESSION', session_name, flush=True)
s = state.get_session(session_name)
r = ColabRuntime(s.url, s.token)
code = '''
import subprocess, time, re
from pathlib import Path
print('SERVER_UTC', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), flush=True)
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'], text=True), flush=True)
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'], text=True), flush=True)
root=Path('/content/microduck/logs/rsl_rl/jump_spin_launch_v2')
drive_root=Path('/content/drive/MyDrive/microduck-training/logs/rsl_rl/jump_spin_launch_v2')
drive_checkpoints=list(drive_root.rglob('model_*.pt'))
if drive_checkpoints:
    d=max(drive_checkpoints,key=lambda x:(int(x.stem.split('_')[-1]),x.stat().st_mtime))
    print('LATEST_DRIVE_CHECKPOINT',str(d),'SIZE',d.stat().st_size,'MTIME',d.stat().st_mtime,flush=True)
for index in range(2):
    paths=list(root.rglob('model_*.pt'))
    if paths:
        p=max(paths,key=lambda x:x.stat().st_mtime)
        print('LATEST_LOCAL_CHECKPOINT',str(p), 'MTIME',p.stat().st_mtime,flush=True)
    log=Path('/content/drive/MyDrive/microduck-training/training.log')
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-16000))
            tail=f.read().decode(errors='replace')
        iterations=re.findall(r'Learning iteration\\s+([0-9]+/[0-9]+)',tail)
        print('DRIVE_LOG_LAST_ITERATIONS',iterations[-3:], 'MTIME',log.stat().st_mtime,flush=True)
        print('DRIVE_LOG_TAIL',tail[-600:],flush=True)
    if index==0: time.sleep(10)
'''
try:
    r.execute_code(code, output_hook=display_output, timeout=60)
finally:
    if r.kernel_id and r.kernel_id != s.kernel_id:
        r.stop(shutdown_kernel=True)
    elif r.kernel_id:
        r.stop()

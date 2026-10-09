#!/usr/bin/env python3
"""Explicit opt-in Colab run: seated-pose preflight → 64×5 smoke + ONNX → train → five videos."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

TASK = 'Mjlab-SeatedGreeting-Flat-MicroDuck'
EXPERIMENT = 'seated_greeting_v1'


def latest_checkpoint(root):
    files=[p for p in root.rglob('model_*.pt') if re.fullmatch(r'model_\d+\.pt',p.name)]
    return max(files,key=lambda p:(int(p.stem.split('_')[-1]),p.stat().st_mtime)) if files else None


def export_verified(checkpoint, output, min_iteration):
    import torch
    import mjlab_microduck.jump_spin_plugin
    from mjlab_microduck.export import ExportConfig, run_export
    from mjlab_microduck.publish.manifest import check_onnx, smoke_run_onnx
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if int(payload['iter']) < min_iteration: raise RuntimeError('Checkpoint is earlier than the requested target')
    if not all(bool(torch.isfinite(v).all()) for v in payload['actor_state_dict'].values() if torch.is_tensor(v)):
        raise RuntimeError('Non-finite actor checkpoint')
    run_export(TASK,ExportConfig(checkpoint_file=str(checkpoint),onnx_file=str(output),num_envs=1,device='cuda:0'))
    check_onnx(output)
    smoke_run_onnx(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-training',action='store_true',help='Explicitly start GPU work')
    parser.add_argument('--target-iters',type=int,default=3000)
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--drive-root',type=Path,default=Path('/content/drive/MyDrive/microduck-training/seated_greeting_v1'))
    args=parser.parse_args()
    if not args.start_training:
        print('준비 모드입니다. 학습을 시작하려면 내일 --start-training을 지정하세요.');return
    if args.target_iters<5 or args.num_envs<1: parser.error('target-iters >= 5 and num-envs >= 1 required')
    if not Path('/content/drive/MyDrive').is_dir(): raise RuntimeError('Mount Google Drive first')
    repo=Path(__file__).resolve().parents[1]
    args.drive_root=args.drive_root.resolve()
    args.drive_root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,WANDB_MODE='offline',WANDB_SILENT='true',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
    env.pop('MICRODUCK_WARM_START',None)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=args.drive_root/'evaluations'/stamp
    output.mkdir(parents=True)

    def status(state,**extra):
        (args.drive_root/'status.json').write_text(json.dumps(dict(state=state,updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2))

    def run(cmd):
        print('RUN:', ' '.join(map(str,cmd)),flush=True)
        subprocess.run(list(map(str,cmd)),cwd=repo,env=env,check=True)

    try:
        status('preflight')
        run(['uv','run','python','scripts/evaluate_seated_greeting.py','--preflight','--output-dir',output])
        status('smoke_test')
        smoke_run='greeting-smoke-'+stamp
        run(['uv','run','train',TASK,'--env.scene.num-envs','64','--agent.max_iterations','5',
             '--agent.experiment-name','seated_greeting_smoke','--agent.run-name',smoke_run])
        candidates=list((repo/'logs/rsl_rl/seated_greeting_smoke').glob('*'+smoke_run+'/model_4.pt'))
        if len(candidates)!=1: raise RuntimeError('Fresh smoke checkpoint was not found uniquely')
        export_verified(candidates[0],output/'smoke_policy.onnx',4)
        # Only this task's isolated backup folder is restored; never use a JumpSpin checkpoint.
        local=repo/'logs/rsl_rl'/EXPERIMENT
        backup=args.drive_root/'logs/rsl_rl'/EXPERIMENT
        local.mkdir(parents=True,exist_ok=True)
        if backup.exists(): run(['rsync','-a',str(backup)+'/',str(local)+'/'])
        checkpoint=latest_checkpoint(local)
        import torch
        reached=checkpoint is not None and int(torch.load(checkpoint,map_location='cpu',weights_only=False)['iter'])>=args.target_iters-1
        if not reached:
            status('training',target_iterations=args.target_iters)
            env.update(TASK_ID=TASK,EXPERIMENT_NAME=EXPERIMENT,NUM_ENVS=str(args.num_envs),
                       TARGET_ITERS=str(args.target_iters),DRIVE_ROOT=str(args.drive_root),REPO_DIR=str(repo),
                       SYNC_INTERVAL='10',SMOKE_TEST='0',RUN_NAME='seated-greeting')
            run(['bash','scripts/train_colab.sh'])
        checkpoint=latest_checkpoint(local)
        if checkpoint is None: raise RuntimeError('No final checkpoint')
        status('exporting',checkpoint=str(checkpoint))
        export_verified(checkpoint,output/'policy.onnx',args.target_iters-1)
        status('evaluating',checkpoint=str(checkpoint),output_dir=str(output))
        run(['uv','run','python','scripts/evaluate_seated_greeting.py','--checkpoint',checkpoint,'--output-dir',output,'--trials','5'])
        summary=json.loads((output/'evaluation_summary.json').read_text())
        if summary['videos_created']!=5 or any(not Path(t['video']).is_file() or Path(t['video']).stat().st_size==0 for t in summary['trials']):
            raise RuntimeError('Five complete videos were not saved')
        status('completed',checkpoint=str(checkpoint),output_dir=str(output),sim_checks_passed=summary['all_sim_checks_passed'])
        print('결과 저장:',output,flush=True)
        print('시뮬레이션 검사 통과:',summary['all_sim_checks_passed'],'— 실제 영상 확인이 필요합니다.',flush=True)
    except BaseException as exc:
        status('failed',error=str(exc),output_dir=str(output));raise


if __name__=='__main__': main()

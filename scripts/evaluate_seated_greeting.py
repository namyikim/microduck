#!/usr/bin/env python3
"""Preflight seated BAM poses or render policy greetings, without hiding failed trials."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab_microduck.jump_spin_plugin  # registers both local experiments
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_seated_greeting_env_cfg import TASK_ID
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES
from mjlab_microduck.seated_greeting import DURATION_S, greeting_pose, assess_greeting


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--width',type=int,default=480)
    parser.add_argument('--height',type=int,default=480)
    args=parser.parse_args()
    if not args.preflight and not args.checkpoint:
        parser.error('--checkpoint is required for policy evaluation')
    if args.trials < 1: parser.error('--trials must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=load_env_cfg(TASK_ID,play=True)
    cfg.scene.num_envs=64 if args.preflight else 1
    # Take the final sample before timeout/reset; never omit terminal contact evidence.
    cfg.episode_length_s=DURATION_S+5
    cfg.terminations.pop('greeting_failed')
    cfg.viewer.width=args.width;cfg.viewer.height=args.height
    # Frame the seated robot closely enough to inspect the small head gestures.
    cfg.viewer.distance=.7
    cfg.viewer.lookat=(0.,0.,.04)
    cfg.viewer.elevation=-15.
    if args.preflight:
        cfg.events['set_ground_state'].params.update(sitting_joint_noise_std=.02,sitting_tilt_max=math.radians(2))
    raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode=None if args.preflight else 'rgb_array')
    agent=load_rl_cfg(TASK_ID)
    env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
    robot=raw.scene['robot']
    servo_ids=mdp._servo_joint_ids(raw,robot)
    head_ids,_=robot.find_joints_by_actuator_names(mdp._NECK_JOINT_PATTERNS)
    result={'task':TASK_ID,'checkpoint':str(args.checkpoint) if args.checkpoint else None,
            'kind':'preflight' if args.preflight else 'policy_evaluation','trials':[]}
    try:
        if args.preflight:
            obs=env.get_observations()
            target=mdp._servo_default_joint_pos(raw,robot).clone()
            for i,v in SITTING_TARGET_OVERRIDES.items(): target[:,i]=v
            # Hold the seated target for 3 seconds, then exercise the complete head sequence.
            max_tilt=0.;head_frames=0;bad_height_frames=0
            for step in range(round((DURATION_S+3)/raw.step_dt)):
                t=step*raw.step_dt
                pose=greeting_pose(max(0,t-3))
                for col,j in enumerate(head_ids):
                    target[:,servo_ids.index(j)]=robot.data.default_joint_pos[:,j]+pose[col]
                actions=target-mdp._servo_default_joint_pos(raw,robot)
                _,_,dones,_=env.step(actions)
                if bool(dones.any()): raise RuntimeError('Preflight reset/NaN; do not train')
                q=robot.data.root_link_quat_w
                upright=1-2*(q[:,1]**2+q[:,2]**2)
                z=robot.data.root_link_pos_w[:,2]-raw.scene.terrain.env_origins[:,2]
                max_tilt=max(max_tilt,float(torch.acos(upright.clamp(-1,1)).max())*180/math.pi)
                head_frames+=int(mdp.seated_greeting_head_contact_cost(raw).sum())
                bad_height_frames+=int(((z<.04)|(z>.085)|~torch.isfinite(z)).sum())
            result.update(max_tilt_deg=max_tilt,head_contact_frames=head_frames,bad_height_frames=bad_height_frames)
            result['passed']=max_tilt<=20 and head_frames==0 and bad_height_frames==0
            (args.output_dir/'preflight.json').write_text(json.dumps(result,indent=2))
            if not result['passed']: raise RuntimeError('Seated pose/head sweep failed preflight; inspect preflight.json before training')
        else:
            import imageio.v2 as imageio
            runner=load_runner_cls(TASK_ID)(env,asdict(agent),device='cuda:0')
            runner.load(str(args.checkpoint),load_cfg={'actor':True},strict=True,map_location='cuda:0')
            policy=runner.get_inference_policy(device='cuda:0')
            for trial in range(1,args.trials+1):
                obs,_=env.reset()
                rows=[];video=args.output_dir/f'seated_greeting_trial_{trial:02d}.mp4'
                with imageio.get_writer(str(video),fps=10) as writer:
                    for step in range(round(DURATION_S/raw.step_dt)+1):
                        t=step*raw.step_dt
                        q=robot.data.root_link_quat_w[0]
                        actual=(robot.data.joint_pos[0,head_ids]-robot.data.default_joint_pos[0,head_ids]).detach().cpu().tolist()
                        command=raw.command_manager.get_command('head_pose')[0].detach().cpu().tolist()
                        rows.append(dict(time_s=t,height_m=float(robot.data.root_link_pos_w[0,2]-raw.scene.terrain.env_origins[0,2]),
                                         upright_cos=float(1-2*(q[1]**2+q[2]**2)),
                                         head_contact=bool(mdp.seated_greeting_head_contact_cost(raw)[0]),
                                         head_actual=actual,head_target=command,
                                         head_error_rad=max(abs(a-b) for a,b in zip(actual,command))))
                        if step%5==0: writer.append_data(raw.render())
                        if step==round(DURATION_S/raw.step_dt): break
                        # Delay buffers must remain mutable by the next trial's reset.
                        with torch.no_grad(): obs,_,dones,_=env.step(policy(obs))
                        if bool(dones.any()): break
                assessment=assess_greeting(rows)
                assessment.update(trial=trial,video=str(video))
                result['trials'].append(assessment)
                (args.output_dir/f'seated_greeting_trial_{trial:02d}_trace.json').write_text(json.dumps(rows,indent=2))
                print(json.dumps(assessment),flush=True)
            result['videos_created']=len(result['trials'])
            result['all_sim_checks_passed']=all(t['sim_checks_passed'] for t in result['trials'])
            (args.output_dir/'evaluation_summary.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
    finally:
        env.close()


if __name__=='__main__': main()

"""Small head-command sequence and conservative simulation checks (no simulator imports)."""
import math

DURATION_S = 14.0
# Seconds, head_pitch delta, head_yaw delta (radians relative to HOME).
# Cosine easing gives zero velocity at every hold/turnaround; no action filtering.
WAYPOINTS = ((0., 0., 0.), (2., 0., 0.), (3., 0., math.radians(15)),
             (4., 0., math.radians(15)), (6., 0., -math.radians(15)),
             (7., 0., -math.radians(15)), (8., 0., 0.),
             (9., math.radians(8), 0.), (10., 0., 0.),
             (11., math.radians(8), 0.), (12., 0., 0.), (14., 0., 0.))


def greeting_pose(time_s):
    """[neck_pitch, head_pitch, head_yaw, head_roll] deltas; neutral outside cycle."""
    for a, b in zip(WAYPOINTS, WAYPOINTS[1:]):
        if a[0] <= time_s < b[0]:
            u = (time_s-a[0]) / (b[0]-a[0])
            w = .5-.5*math.cos(math.pi*u)
            return (0., a[1]+w*(b[1]-a[1]), a[2]+w*(b[2]-a[2]), 0.)
    return (0., 0., 0., 0.)


def assess_greeting(samples):
    """Checks full sequence, posture and each gesture separately; never certifies hardware."""
    reasons = []
    if not samples:
        return {'sim_checks_passed': False, 'reasons': ['no_samples']}
    fields = ('time_s', 'height_m', 'upright_cos', 'head_error_rad')
    if any(s.get('head_contact') not in (True, False) or s.get('head_contact') is None
           or any(not isinstance(s.get(k), (int,float)) or not math.isfinite(s[k]) for k in fields)
           for s in samples):
        return {'sim_checks_passed': False, 'reasons': ['missing_or_nonfinite_evidence']}
    for s in samples:
        for key in ('head_actual','head_target'):
            values=s.get(key)
            if not isinstance(values,(list,tuple)) or len(values)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
                return {'sim_checks_passed': False, 'reasons': ['missing_or_nonfinite_head_evidence']}
    if any(max(abs(a-b) for a,b in zip(s['head_target'],greeting_pose(s['time_s'])))>.025 for s in samples):
        reasons.append('incorrect_command_sequence')
    if samples[0]['time_s'] > .021 or samples[-1]['time_s'] < DURATION_S-.05:
        reasons.append('incomplete_sequence')
    if any(b['time_s'] <= a['time_s'] or b['time_s']-a['time_s'] > .041
           for a,b in zip(samples,samples[1:])):
        reasons.append('missing_frames')
    if any(s['head_contact'] for s in samples):
        reasons.append('head_ground_contact')
    if any(s['upright_cos'] < math.cos(math.radians(20)) for s in samples):
        reasons.append('excessive_trunk_tilt')
    if any(not .04 <= s['height_m'] <= .085 for s in samples):
        reasons.append('not_seated')
    phase_errors = {}
    for name,lo,hi in [('left',3,4),('right',6,7),('nod_1',8.8,9.2),('nod_2',10.8,11.2),('rest',12.5,13.9)]:
        group = [max(abs(a-b) for a,b in zip(s['head_actual'],s['head_target'])) for s in samples if lo <= s['time_s'] <= hi]
        phase_errors[name] = sum(group)/len(group) if group else None
        if not group or phase_errors[name] > .06:
            reasons.append('tracking_'+name)
    return {'sim_checks_passed': not reasons, 'reasons': reasons,
            'phase_mean_max_joint_error_rad': phase_errors,
            'max_tilt_deg': math.degrees(math.acos(max(-1.,min(1.,min(s['upright_cos'] for s in samples))))),
            'head_contact_frames': sum(s['head_contact'] for s in samples),
            'note': 'Simulation screening only; inspect videos before considering hardware.'}

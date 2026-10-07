import math

from mjlab_microduck.tasks.microduck_jump_spin_env_cfg import (
    TARGET_FINAL,
    TARGET_FINAL_STEP,
    TARGET_MIDDLE,
    TARGET_MIDDLE_STEP,
    MicroduckJumpSpinRlCfg,
    make_microduck_jump_spin_env_cfg,
)


def test_jump_spin_cfg_has_core_rewards():
    cfg = make_microduck_jump_spin_env_cfg()
    for name in (
        "jump_launch",
        "jump_height",
        "jump_spin_progress",
        "jump_airborne_upright",
        "jump_landing",
        "jump_heading_return",
        "jump_wrong_axis",
        "jump_horizontal_drift",
        "jump_landing_impact",
    ):
        assert name in cfg.rewards, name

    assert cfg.rewards["jump_spin_progress"].weight > 0.0
    assert cfg.rewards["jump_landing"].weight > 0.0
    assert cfg.rewards["jump_wrong_axis"].weight < 0.0
    assert cfg.rewards["jump_landing_impact"].weight < 0.0


def test_jump_spin_target_curriculum_reaches_360():
    assert TARGET_MIDDLE == 1.5 * math.pi
    assert TARGET_FINAL == 2.0 * math.pi
    assert TARGET_MIDDLE_STEP == 1500 * 24
    assert TARGET_FINAL_STEP == 4000 * 24


def test_jump_spin_reverse_curriculum_shifts_to_real_starts():
    cfg = make_microduck_jump_spin_env_cfg()
    stages = cfg.curriculum["jump_spin_spawn_mix"].params["param_stages"]
    assert stages[0]["params"]["airborne_prob"] > stages[0]["params"]["standing_prob"]
    assert stages[-1]["params"]["standing_prob"] >= 0.85
    assert stages[-1]["params"]["airborne_prob"] <= 0.10


def test_jump_spin_play_mode_always_starts_standing():
    cfg = make_microduck_jump_spin_env_cfg(play=True)
    params = cfg.events["set_jump_spin_state"].params
    assert params["standing_prob"] == 1.0
    assert params["airborne_prob"] == 0.0
    assert params["landing_prob"] == 0.0


def test_jump_spin_keeps_61d_observation_layout():
    from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
        make_microduck_roulade_env_cfg,
    )

    jump = make_microduck_jump_spin_env_cfg()
    roulade = make_microduck_roulade_env_cfg()
    for group in ("actor", "critic"):
        assert list(jump.observations[group].terms.keys()) == list(
            roulade.observations[group].terms.keys()
        )


def test_jump_spin_direction_specific_symmetry_is_disabled():
    assert MicroduckJumpSpinRlCfg.algorithm.symmetry_cfg is None


def test_jump_spin_saves_more_frequently_for_colab():
    assert MicroduckJumpSpinRlCfg.save_interval == 100
    assert MicroduckJumpSpinRlCfg.max_iterations == 10_000

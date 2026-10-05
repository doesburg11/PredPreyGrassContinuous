"""Training configuration regressions, without starting Ray workers."""

import math
from pathlib import Path

import pytest

import train


def test_learning_defaults():
    env, args = train.load_settings()
    assert args.num_env_runners == 16
    assert args.num_gpus_per_learner == 1
    assert args.train_batch_size == 4800
    assert args.minibatch_size == 1024
    # Fixed by grass/energy, so make_env sets them (see test below).
    assert "obs_mode" not in env and "keep_prey_count_constant" not in env
    assert env["reward_scale"] == 1.0
    assert args.entropy_coeff == 0.01
    assert args.lr == 3e-4
    assert args.gae_lambda == 0.95
    assert args.num_epochs == 10
    assert args.grad_clip == 0.5
    assert not args.vf_share_layers
    # PredPreyGrass's sparse rewards: births only.
    assert env["prey_reward"] == env["prey_punishment"] == 0
    assert env["predator_reward"] == env["grass_food_reward"] == 0
    assert env["reproduction_predator_reward"] == 10.0
    assert env["reproduction_prey_reward"] == 10.0
    assert args.mode == "ps"


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True])
def test_reward_scale_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setitem(train.config_env, "reward_scale", value)
    with pytest.raises(ValueError, match="reward_scale"):
        train.load_settings()


@pytest.mark.parametrize("name", ["gamma", "gae_lambda"])
@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_discount_validation(monkeypatch, name, value):
    monkeypatch.setitem(train.config_ppo, name, value)
    with pytest.raises(ValueError, match=name):
        train.load_settings()


@pytest.mark.parametrize(
    "name", ["num_env_runners", "num_learners", "checkpoint_every"]
)
def test_resource_counts_reject_negative_values(monkeypatch, name):
    monkeypatch.setitem(train.config_ppo, name, -1)
    with pytest.raises(ValueError, match=name):
        train.load_settings()


def test_checkpoint_configuration(monkeypatch, tmp_path):
    monkeypatch.setitem(train.config_ppo, "checkpoint_every", 10)
    monkeypatch.setitem(train.config_ppo, "checkpoint_dir", None)
    with pytest.raises(ValueError, match="checkpoint_every"):
        train.load_settings()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(train.config_ppo, "checkpoint_dir", "checkpoint")
    _, args = train.load_settings()
    assert args.checkpoint_dir == str(tmp_path / "checkpoint")
    assert train.config_ppo["checkpoint_dir"] == "checkpoint"


def test_settings_are_copied_and_shaping_discount_is_synchronized(monkeypatch):
    monkeypatch.setitem(train.config_ppo, "gamma", 0.9)
    env, args = train.load_settings()
    assert env["shaping_gamma"] == args.gamma == 0.9
    env["prey_count"] = 99
    args.lr = 1
    assert train.config_env["prey_count"] == 8
    assert train.config_ppo["lr"] == 3e-4


def test_gradient_clipping_can_be_disabled(monkeypatch):
    monkeypatch.setitem(train.config_ppo, "grad_clip", None)
    assert train.load_settings()[1].grad_clip is None


def test_cpu_only_override(monkeypatch):
    monkeypatch.setitem(train.config_ppo, "num_env_runners", 0)
    monkeypatch.setitem(train.config_ppo, "num_gpus_per_learner", 0)
    args = train.load_settings()[1]
    assert args.num_env_runners == args.num_gpus_per_learner == 0


def test_file_settings_reach_ppo_and_environment(monkeypatch):
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(variable, "original")
    thread_limits = []
    monkeypatch.setattr(train.torch, "set_num_threads", thread_limits.append)
    captured = []

    class ConfigCaptured(Exception):
        pass

    def capture(config):
        captured.append(config)
        raise ConfigCaptured

    monkeypatch.setattr(train.PPOConfig, "build_algo", capture)
    monkeypatch.setitem(train.config_ppo, "gamma", 0.9)
    monkeypatch.setitem(train.config_ppo, "vf_clip_param", 2000)
    monkeypatch.setitem(train.config_env, "predator_shaping", 1)
    monkeypatch.setitem(train.config_env, "reward_scale", 0.02)
    monkeypatch.setitem(train.config_env, "keep_prey_count_constant", False)
    monkeypatch.setitem(train.config_env, "prey_fov", 360)
    monkeypatch.setitem(train.config_env, "predator_count", 2)
    monkeypatch.setitem(train.config_env, "prey_count", 3)
    monkeypatch.setitem(train.config_ppo, "mode", "ps")
    with pytest.raises(ConfigCaptured):
        train.main()
    config = captured[0]
    assert config.gamma == config.env_config["shaping_gamma"] == 0.9
    assert config.env_config["reward_scale"] == 0.02
    assert config.env_config["predator_shaping"] == 1
    assert not config.env_config["keep_prey_count_constant"]
    assert config.env_config["prey_fov"] == 360
    assert set(config.policies) == {"predator_policy", "prey_policy"}
    assert config.vf_clip_param == 2000
    assert config.grad_clip == 0.5
    assert not config.model_config["vf_share_layers"]
    assert thread_limits == [1]
    assert config.num_env_runners == 16
    assert config.num_cpus_per_env_runner == 1
    assert config.num_cpus_for_main_process == 1
    assert config.num_gpus_per_learner == 1
    assert config.train_batch_size_per_learner == 4800
    assert config.minibatch_size == 1024
    import os

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[variable] == "1"
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(
        Path(train.__file__).parent
    )


def test_training_iterations_checkpoints_and_logging(monkeypatch, tmp_path):
    events = []

    class FakeAlgorithm:
        def train(self):
            events.append("train")
            return {"env_runners": {"episode_return_mean": 1.0}}

        def save(self, path):
            events.append(("save", path))

        def stop(self):
            events.append("stop")

    class FakeWriter:
        def __init__(self, path):
            events.append(("writer", path))

        def add_scalar(self, tag, value, step):
            events.append(("scalar", tag, value, step))

        def flush(self):
            pass

        def close(self):
            events.append("close_writer")

    checkpoint = str(tmp_path / "checkpoint")
    monkeypatch.setattr(train.PPOConfig, "build_algo", lambda self: FakeAlgorithm())
    monkeypatch.setattr(train, "SummaryWriter", FakeWriter)
    monkeypatch.setattr(train, "register", lambda: None)
    monkeypatch.setattr(train.torch, "set_num_threads", lambda n: None)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(name, "original")
    for key, value in {
        "iterations": 3,
        "checkpoint_every": 2,
        "checkpoint_dir": checkpoint,
        "tensorboard_dir": "tb",
    }.items():
        monkeypatch.setitem(train.config_ppo, key, value)
    train.main()
    assert events.count("train") == 3
    assert [
        event for event in events if isinstance(event, tuple) and event[0] == "save"
    ] == [("save", str(tmp_path / "checkpoint" / "iter_000002")), ("save", checkpoint)]
    assert [
        event[-1]
        for event in events
        if isinstance(event, tuple) and event[0] == "scalar"
    ] == [1, 2, 3]
    assert events[-2:] == ["close_writer", "stop"]


def test_output_paths_share_module_name_and_timestamp(monkeypatch, tmp_path):
    import re

    for key, folder in (
        ("checkpoint_dir", "checkpoint"),
        ("tensorboard_dir", "tensorboard"),
    ):
        monkeypatch.setitem(
            train.config_ppo, key, str(tmp_path / "{run_name}" / folder)
        )
    _, settings = train.load_settings()
    checkpoint = Path(settings.checkpoint_dir)
    tensorboard = Path(settings.tensorboard_dir)
    assert checkpoint.parent == tensorboard.parent
    assert checkpoint.parent.parent == tmp_path
    assert re.fullmatch(
        r"add_reproduction_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}",
        checkpoint.parent.name,
    )
    assert "{run_name}" in train.config_ppo["checkpoint_dir"]


def test_individual_policies_are_rejected_with_reproduction(monkeypatch):
    monkeypatch.setitem(train.config_ppo, "mode", "il")
    with pytest.raises(ValueError, match="il"):
        train.load_settings()


def test_make_env_fills_in_required_obs_mode_and_permanent_deaths():
    from env_wrapper import make_env

    env_config, _ = train.load_settings()
    env = make_env(env_config)
    try:
        raw = env.par_env.aec_env.unwrapped
        assert raw.keep_prey_count_constant is False
        assert raw._egocentric_obs_patched
        assert env.get_observation_space("prey_0").shape == (33,)
    finally:
        env.close()

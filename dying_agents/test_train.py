"""Training configuration regressions, without starting Ray workers."""

import pytest

from train import build_parser


def test_learning_defaults():
    args = build_parser().parse_args([])
    assert args.num_env_runners == 16
    assert args.num_gpus_per_learner == 1
    assert args.train_batch_size == 4800
    assert args.minibatch_size == 1024
    assert args.obs == "egocentric"
    assert args.reward_scale == 0.01
    assert args.entropy_coeff == 0.01
    assert args.lr == 3e-4
    assert args.gae_lambda == 0.95
    assert args.num_epochs == 10
    assert args.grad_clip == 0.5
    assert not args.vf_share_layers
    # Aquarium's death penalty becomes -10, rather than -1000.
    assert args.vf_clip_param >= 10 * (1000 * args.reward_scale) ** 2


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_reward_scale_rejects_invalid_values(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--reward-scale", value])


def test_learning_settings_can_be_overridden():
    args = build_parser().parse_args(
        [
            "--obs",
            "aquarium",
            "--reward-scale",
            "1",
            "--entropy-coeff",
            "0",
            "--lr",
            "5e-5",
            "--gae-lambda",
            "1",
            "--num-epochs",
            "30",
            "--vf-share-layers",
            "--gamma",
            "0.9",
            "--vf-clip-param",
            "1000",
        ]
    )
    assert args.obs == "aquarium"
    assert args.reward_scale == 1
    assert args.vf_share_layers
    assert args.gamma == 0.9
    assert args.vf_clip_param == 1000

    assert args.entropy_coeff == 0
    assert args.lr == 5e-5
    assert args.gae_lambda == 1
    assert args.num_epochs == 30


@pytest.mark.parametrize("flag", ["--gamma", "--gae-lambda"])
@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_discount_validation(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, value])


def test_gradient_clipping_can_be_disabled():
    assert build_parser().parse_args(["--no-grad-clip"]).grad_clip is None
    assert build_parser().parse_args(["--grad-clip", "2"]).grad_clip == 2


def test_cli_settings_reach_ppo_and_environment(monkeypatch):
    import train

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
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--gamma",
            "0.9",
            "--predator-shaping",
            "1",
            "--reward-scale",
            "0.02",
            "--vf-clip-param",
            "2000",
        ],
    )
    with pytest.raises(ConfigCaptured):
        train.main()
    config = captured[0]
    assert config.gamma == config.env_config["shaping_gamma"] == 0.9
    assert config.env_config["reward_scale"] == 0.02
    assert config.env_config["predator_shaping"] == 1
    assert config.env_config["obs_mode"] == "egocentric"
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
    from pathlib import Path

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[variable] == "1"
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(
        Path(train.__file__).parent
    )


def test_cpu_only_override():
    args = build_parser().parse_args(
        [
            "--num-env-runners",
            "0",
            "--num-gpus-per-learner",
            "0",
        ]
    )
    assert args.num_env_runners == args.num_gpus_per_learner == 0

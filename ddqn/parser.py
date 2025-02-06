import argparse

ddqn_parser = argparse.ArgumentParser("ddqn")

ddqn_parser.add_argument(
    "--buffer-size",
    help="Max size of the experience replay buffer.",
    type=int,
    default=10_000,
)
ddqn_parser.add_argument(
    "--num-episodes",
    help="Number of episodes to train for.",
    type=int,
    default=600,
)
ddqn_parser.add_argument(
    "--batch-size",
    help="Batch size to be used in training.",
    type=int,
    default=128,
)
ddqn_parser.add_argument(
    "--gamma",
    help="Discounting power γ of past actions in computing reward",
    type=float,
    default=0.995,
)
ddqn_parser.add_argument(
    "--alpha",
    help="Learing rate α of the optimiser to be used in training.",
    type=float,
    default=1e-5,
)
ddqn_parser.add_argument(
    "--tau",
    help="Update rate τ of target network (soft update).",
    type=float,
    default=1e-3,
)
ddqn_parser.add_argument(
    "--outdir",
    help="Directory to save results to. Will be created if it does not exist.",
    type=str,
    default="./out",
)
ddqn_parser.add_argument(
    "--eval",
    help="Runs the model specified in --outdir and renders the run",
    type=bool,
    action=argparse.BooleanOptionalAction,
)
ddqn_parser.add_argument(
    "--save-path",
    help="If supplied with the --eval flag set, will save a GIF of a run to the supplied path",  # type:ignore
    type=str,
    default=None,
)
ddqn_parser.add_argument(
    "--checkpoint-episodes",
    help="Frequency with which to save checkpoints",  # type:ignore
    type=int,
    default=1000,
)

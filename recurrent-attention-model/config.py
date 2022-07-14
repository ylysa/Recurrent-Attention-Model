import argparse

arg_lists = []
parser = argparse.ArgumentParser(description="RAM")


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# question network params

question_arg = add_argument_group("Question Network Params")
question_arg.add_argument(
    "--qa_hidden", type=int, default=128, help="hidden size of question-answer fc"
)
question_arg.add_argument(
    "--img_hidden", type=int, default=128, help="hidden size of image fc"
)

# core network params
core_arg = add_argument_group("Core Network Params")
core_arg.add_argument("--hidden_size", type=int, default=256, help="hidden size of rnn")


# reinforce params
reinforce_arg = add_argument_group("Reinforce Params")
reinforce_arg.add_argument(
    "--std", type=float, default=0.05, help="gaussian policy standard deviation"
)

reinforce_arg.add_argument(
    "--M", type=int, default=1, help="Monte Carlo sampling for valid and test sets"
)

# training params
train_arg = add_argument_group("Training Params")
train_arg.add_argument(
    "--is_train", type=str2bool, default=True, help="Whether to train or test the model"
)
train_arg.add_argument(
    "--momentum", type=float, default=0.5, help="Nesterov momentum value"
)
train_arg.add_argument(
    "--epochs", type=int, default=300, help="# of epochs to train for"
)
train_arg.add_argument(
    "--pretraining_epochs", type=int, default=25, help="# of pretraining epochs to train for"
)
train_arg.add_argument(
    "--init_lr", type=float, default=3e-4, help="Initial learning rate value"
)
train_arg.add_argument(
    "--lr_patience",
    type=int,
    default=20,
    help="Number of epochs to wait before reducing lr",
)
train_arg.add_argument(
    "--train_patience",
    type=int,
    default=100,
    help="Number of epochs to wait before stopping train",
)


# other params
misc_arg = add_argument_group("Misc.")
misc_arg.add_argument(
    "--use_gpu", type=str2bool, default=True, help="Whether to run on the GPU"
)
misc_arg.add_argument(
    "--best",
    type=str2bool,
    default=True,
    help="Load best model or most recent for testing",
)
misc_arg.add_argument(
    "--random_seed", type=int, default=1, help="Seed to ensure reproducibility"
)
misc_arg.add_argument(
    "--data_dir", type=str, default="./data", help="Directory in which data is stored"
)
misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default="./ckpt",
    help="Directory in which to save model checkpoints",
)
misc_arg.add_argument(
    "--logs_dir",
    type=str,
    default="./logs/",
    help="Directory in which Tensorboard logs wil be stored",
)
misc_arg.add_argument(
    "--resume",
    type=str2bool,
    default=False,
    help="Whether to resume training from checkpoint",
)
misc_arg.add_argument(
    "--print_freq",
    type=int,
    default=10,
    help="How frequently to print training details",
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

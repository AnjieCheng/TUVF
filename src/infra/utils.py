import os
import math
import shutil
import subprocess
from distutils.dir_util import copy_tree
from shutil import copyfile
from typing import List, Optional
from rich.console import Console
import logging

from hydra.utils import instantiate
import click
import git
from omegaconf import DictConfig

logging.disable(logging.CRITICAL)
CONSOLE = Console(width=180)

# ----------------------------------------------------------------------------


def copy_objects(target_dir: os.PathLike, objects_to_copy: List[os.PathLike]):
    for src_path in objects_to_copy:
        trg_path = os.path.join(target_dir, os.path.basename(src_path))

        if os.path.islink(src_path):
            os.symlink(os.readlink(src_path), trg_path)
        elif os.path.isfile(src_path):
            copyfile(src_path, trg_path)
        elif os.path.isdir(src_path):
            copy_tree(src_path, trg_path)
        else:
            raise NotImplementedError(f"Unknown object type: {src_path}")


# ----------------------------------------------------------------------------


def create_symlinks(target_dir: os.PathLike, symlinks_to_create: List[os.PathLike]):
    for src_path in symlinks_to_create:
        trg_path = os.path.join(target_dir, os.path.basename(src_path))

        if os.path.islink(src_path):
            os.symlink(os.readlink(src_path), trg_path)
        else:
            CONSOLE.log(
                f"Creating a symlink to {src_path}, so try not to delete it occasionally!"
            )
            os.symlink(src_path, trg_path)


# ----------------------------------------------------------------------------


def is_git_repo(path: os.PathLike):
    try:
        _ = git.Repo(path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


# ----------------------------------------------------------------------------


def create_project_dir(
    project_dir: os.PathLike,
    objects_to_copy: List[os.PathLike],
    symlinks_to_create: List[os.PathLike],
    quiet: bool = False,
    ignore_uncommited_changes: bool = False,
    overwrite: bool = False,
):
    if is_git_repo(os.getcwd()) and are_there_uncommitted_changes():
        if ignore_uncommited_changes or click.confirm(
            "There are uncommited changes. Continue?", default=False
        ):
            pass
        else:
            raise PermissionError(
                "Cannot created a dir when there are uncommited changes"
            )

    if os.path.exists(project_dir):
        if overwrite or click.confirm(
            f"Dir {project_dir} already exists. Overwrite it?", default=False
        ):
            shutil.rmtree(project_dir)
        else:
            CONSOLE.log("User refused to delete an existing project dir.")
            raise PermissionError("There is an existing dir and I cannot delete it.")

    os.makedirs(project_dir)
    copy_objects(project_dir, objects_to_copy)

    if not quiet:
        CONSOLE.log(f"Created a project dir: {project_dir}")


# ----------------------------------------------------------------------------


def get_git_hash() -> Optional[str]:
    if not is_git_repo(os.getcwd()):
        return None

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except:
        return None


# ----------------------------------------------------------------------------


def get_git_hash_suffix() -> str:
    git_hash: Optional[str] = get_git_hash()
    git_hash_suffix = "-nogit" if git_hash is None else f"-{git_hash}"

    return git_hash_suffix


# ----------------------------------------------------------------------------


def are_there_uncommitted_changes() -> bool:
    return len(subprocess.check_output("git status -s".split()).decode("utf-8")) > 0


# ----------------------------------------------------------------------------


def cfg_to_args_str(cfg: DictConfig, use_dashes=True) -> str:
    dashes = "--" if use_dashes else ""
    return " ".join([f"{dashes}{p}={cfg[p]}" for p in cfg])


# ----------------------------------------------------------------------------


def recursive_instantiate(cfg: DictConfig):
    for key in cfg:
        if isinstance(cfg[key], DictConfig):
            if "_target_" in cfg[key]:
                cfg[key] = instantiate(cfg[key])
            else:
                recursive_instantiate(cfg[key])


# ----------------------------------------------------------------------------


def num_gpus_to_mem(num_gpus: int, mem_per_gpu: 64) -> str:
    # Doing it here since hydra config cannot do formatting for ${...}
    return f"{num_gpus * mem_per_gpu}G"


# ----------------------------------------------------------------------------


def product(values):
    import numpy as np

    return np.prod([x for x in values]).item()


# ----------------------------------------------------------------------------


def divide(dividend, divisor):
    return dividend / divisor


# ----------------------------------------------------------------------------


def log2_divide(dividend, divisor):
    return int(math.log2(dividend / divisor))


# ----------------------------------------------------------------------------


def linspace(val_from: float, val_to: float, num_steps: int) -> List[float]:
    # import numpy as np
    # return np.linspace(val_from, val_to, num_steps).tolist()
    assert num_steps > 1, f"Too small num_steps: {num_steps}"
    return [
        val_from + (val_to - val_from) * i / (num_steps - 1) for i in range(num_steps)
    ]


# ----------------------------------------------------------------------------

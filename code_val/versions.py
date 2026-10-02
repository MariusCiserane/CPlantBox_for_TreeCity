"""Le commit courant, note dans chaque fichier ecrit pour savoir quel code l'a produit.

Le suffixe -dirty signale que code_val/ a ete modifie depuis ce commit : le dataset v2 porte un
commit qui ne contenait pas encore le generateur qui l'a produit, et rien ne le disait. La
verification est limitee a code_val/ et faite une fois par processus : a travers le montage Docker,
un git status du depot entier prend plusieurs minutes.
"""
import subprocess
from functools import cache
from pathlib import Path

UNKNOWN_COMMIT = "inconnu"
DIRTY_SUFFIX = "-dirty"
CODE_DIRECTORY = Path(__file__).resolve().parent


def run_git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True,
                          cwd=CODE_DIRECTORY).stdout.strip()


@cache
def read_commit() -> str:
    try:
        commit = run_git("rev-parse", "--short", "HEAD")
        changes = run_git("status", "--porcelain", "--untracked-files=no", "--", ".")
    except OSError:
        return UNKNOWN_COMMIT
    return (commit or UNKNOWN_COMMIT) + (DIRTY_SUFFIX if changes else "")

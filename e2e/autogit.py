from pathlib import Path
from git import Repo


def git_add_commit_with(message: str):
    repo_path = Path("~/repos/waam-e2e")
    repo = Repo(repo_path)

    # get all files that are different from current HEAD or new
    files = [item.a_path for item in repo.index.diff(None)] + [item.a_path for item in repo.index.diff("HEAD")]

    # also get untracked files:
    files += repo.untracked_files

    # add all files: git add .
    repo.index.add(files)

    # commit all files, don't verify the commit message: git commit -m "..." --no-verify
    repo.index.commit(f"autocommit: {message}", skip_hooks=True)

    # print commited files and hash and message:
    print(f"Committed files: {files}")
    print(f"Commit hash: {repo.head.commit.hexsha}")
    print(f"Commit message: {repo.head.commit.message}")

    # return commit hash:
    return repo.head.commit.hexsha

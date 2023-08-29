from pathlib import Path
from git import Repo


def main():

    # this oath is either in a local machine or remote machine: "~/repos/waam-e2e"
    # execute the following always in local machine

    repo_path = Path("~/repos/waam-e2e")
    repo = Repo(repo_path)

    # get all files that are different from current HEAD or new
    files = [item.a_path for item in repo.index.diff(None)] + [item.a_path for item in repo.index.diff("HEAD")]

    # also get untracked files:
    files += repo.untracked_files

    repo.index.add(files)

    # commit all files, don't verify the commit message
    repo.index.commit("Auto commit from e2e/autogit.py: test", skip_hooks=True)


if __name__ == "__main__":
    main()

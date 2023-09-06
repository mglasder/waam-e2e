from pathlib import Path

import wandb


def download_model(project: str, run_id: str, version: str) -> Path:
    api = wandb.Api()
    artifact_name = f"{project}/model-{run_id}:{version}"
    artifact = api.artifact(f"{artifact_name}")
    artifact_path = artifact.download()
    checkpoint_path = Path(artifact_path) / "model.ckpt"
    return Path(checkpoint_path)

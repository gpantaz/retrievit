from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download


def upload_file_to_hub(
    path_to_local_file: str | Path, path_in_repo: str | Path, repo_id: str, repo_type: str
) -> None:
    """Upload a file to a repository on Hugging Face Hub."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)

    api.upload_file(
        path_or_fileobj=str(path_to_local_file),
        path_in_repo=str(path_in_repo),
        repo_id=repo_id,
        repo_type=repo_type,
    )


def download_nested_folder_from_repo(
    repo_id: str, repo_type: str, subfolder: str, local_dir: str | None = None
) -> None:
    """Download a nested folder from a repository on Hugging Face Hub."""
    fs = HfFileSystem()

    matched_remote_files = fs.find(f"{repo_id}/{subfolder}")
    for remote_file in matched_remote_files:
        remote_subfolder = Path(remote_file).parent

        hf_hub_download(
            repo_id=repo_id,
            filename=Path(remote_file).name,
            local_dir=local_dir,
            repo_type=repo_type,
            subfolder=str(remote_subfolder.relative_to(repo_id)),
        )

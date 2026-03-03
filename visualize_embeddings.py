import argparse
from collections import defaultdict
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tokenizer import Tokenizer, build_vocab_for_task
from tqdm import tqdm

from retrievit.datamodels.datamodels import SpecialTokens, Task
from retrievit.utils.huggingface import download_nested_folder_from_repo


def make_gif(filenames: list[str], movie_path: str, duration: int = 1) -> None:
    """Create a small gif from a list of image files."""
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(movie_path, images, duration=duration)


def build_tokenizer(seq_len: int, task: str = Task.position_retrieval.value) -> Tokenizer:
    """Build tokenizer."""
    tokenizer = Tokenizer(build_vocab_for_task(task=task, position_vocab_size=seq_len))
    return tokenizer


def load_embeddings(
    embedding_path: str, device: str = "cpu", original_seq_len: int = 100, new_seq_len: int = 200
) -> torch.nn.Module:
    """Load embeddings from a file."""
    state_dict = torch.load(embedding_path, map_location=device)
    embedding_weights = state_dict["weight"][:203]
    unset_embeddings = embedding_weights[:-original_seq_len]
    old_embeddings = embedding_weights[-original_seq_len:]
    pos_emb = old_embeddings.unsqueeze(0).permute(0, 2, 1)  # (1, D, N)

    resized = F.interpolate(pos_emb, size=new_seq_len, mode="linear", align_corners=False)

    weights = resized.permute(0, 2, 1).squeeze(0)  # (M, D)

    (_, hidden_size) = state_dict["weight"].shape
    embedding = torch.nn.Embedding(unset_embeddings.shape[0] + weights.shape[0], hidden_size)
    # embedding.load_state_dict(state_dict)
    with torch.no_grad():
        embedding.weight.copy_(torch.concatenate([unset_embeddings, weights]))
    return embedding.to(device=device)


def get_position_token_ids(tokenizer: Tokenizer, seq_len: int) -> torch.Tensor:
    """Get position token IDs."""
    special_tokens = SpecialTokens()
    position_ids = [
        special_tokens.position_token_format.format(index=token_id) for token_id in range(seq_len)
    ]
    return tokenizer(position_ids, return_tensors=True)  # type: ignore[report]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID",
    )

    parser.add_argument(
        "--hf-remote-folder",
        type=str,
        required=True,
        help="Path to the Hugging Face remote folder within the HF repo",
    )

    parser.add_argument(
        "--embedding-local-dirpath",
        type=Path,
        default=Path("storage/models"),
        help="Path to the local directory for storing embeddings",
    )
    parser.add_argument(
        "--plots-directory",
        type=Path,
        default=Path("plots/recallibrate_position"),
        help="Path to the local directory for storing plots",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=200,
        help="Sequence length for the tokenizer",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=2,
        help="Duration of the GIF in seconds",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    embeddings_local_dir = args.embedding_local_dirpath.joinpath(args.hf_remote_folder)
    if not embeddings_local_dir.exists():
        download_nested_folder_from_repo(
            repo_id=args.hf_repo_id,
            subfolder=args.hf_remote_folder,
            local_dir=args.embedding_local_dirpath,
            repo_type="model",
        )

    embedding_paths = sorted(
        embeddings_local_dir.glob("embeddings_*.pt"),
        key=lambda x: int(x.name.split("_")[-1].split(".")[0]),
    )

    tokenizer = build_tokenizer(seq_len=args.seq_len)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_paths = defaultdict(list)
    plots_directory = args.plots_directory.joinpath(args.hf_remote_folder)
    plots_directory.mkdir(parents=True, exist_ok=True)
    for embedding_path in tqdm(embedding_paths):
        step = int(embedding_path.name.split("_")[-1].split(".")[0])
        embedding_model = load_embeddings(embedding_path=embedding_path, device=device)
        position_ids = (
            get_position_token_ids(tokenizer, seq_len=args.seq_len).to(device=device) - 100
        )
        with torch.inference_mode():
            position_embeddings = (
                embedding_model(position_ids.unsqueeze(0)).squeeze(0).cpu().numpy()
            )

        pca = PCA(n_components=2)
        X_r = pca.fit(position_embeddings).transform(position_embeddings)

        fig, ax = plt.subplots()
        og_size = fig.get_size_inches()
        fig.set_size_inches(og_size * 0.80)
        colors = sns.color_palette("rocket_r", X_r.shape[0])
        for row, color in zip(X_r, colors, strict=False):
            plt.scatter(row[0], row[1], color=color, alpha=0.8)

        # Set x/y labels
        plt.xlabel("PCA Component 1", fontsize=12)
        plt.ylabel("PCA Component 2", fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        plt.tight_layout()
        plt.savefig(str(plots_directory.joinpath(f"pca_2D_step{step}_interp.png")))

        all_paths["pca_2D"].append(plots_directory.joinpath(f"pca_2D_step{step}_interp.png"))
        plt.close(fig)

        x = torch.tensor(position_embeddings)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        cosine_sim = torch.mm(x_norm, x_norm.t())

        fig, ax = plt.subplots()

        fig.set_size_inches(fig.get_size_inches() * 0.80)
        fig_path = plots_directory.joinpath(f"cosine_sim_full_step{step}_interp.png")
        sns.heatmap(
            cosine_sim.numpy(),
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]},
        )

        ticks = np.arange(0, cosine_sim.shape[0] + 1, 25)
        ticks[0] = 1
        ticks[-1] = cosine_sim.shape[0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
        ax.tick_params(axis="x", labelsize=12, rotation=0)
        ax.tick_params(axis="y", labelsize=12, rotation=0)
        plt.xlabel("Position Tokens", fontsize=12)
        plt.ylabel("Position Tokens", fontsize=12)

        plt.tight_layout()
        plt.savefig(str(fig_path))

        plt.close(fig)
        all_paths["cosine_sim_full"].append(fig_path)

        for n_components in [16, 32, 64]:
            pca = PCA(n_components=n_components)
            X_r = pca.fit(position_embeddings).transform(position_embeddings)

            x = torch.tensor(X_r)
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            cosine_sim = torch.mm(x_norm, x_norm.t())

            fig, ax = plt.subplots()
            fig.set_size_inches(fig.get_size_inches() * 0.80)
            fig_path = plots_directory.joinpath(
                f"cosine_sim_dim={n_components}_step{step}_interp.png"
            )
            sns.heatmap(
                cosine_sim.numpy(),
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]},
            )

            ticks = np.arange(0, cosine_sim.shape[0] + 1, 25)
            ticks[0] = 1
            ticks[-1] = cosine_sim.shape[0]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks)
            ax.set_yticklabels(ticks)
            ax.tick_params(axis="x", labelsize=12, rotation=0)
            ax.tick_params(axis="y", labelsize=12, rotation=0)
            plt.xlabel("Position Tokens", fontsize=12)
            plt.ylabel("Position Tokens", fontsize=12)

            plt.tight_layout()
            plt.savefig(str(fig_path))

            plt.close(fig)
            all_paths[f"cosine_sim_dim={n_components}"].append(fig_path)

    for key, paths in all_paths.items():
        make_gif(
            paths, movie_path=str(plots_directory.joinpath(f"{key}.gif")), duration=args.duration
        )

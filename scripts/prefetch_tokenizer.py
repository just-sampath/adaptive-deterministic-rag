"""Install OpenAI's verified o200k_base vocabulary for offline application startup."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"


def prefetch(client: httpx.Client) -> None:
    """Publish a complete, verified vocabulary, preserving concurrent readers and installers."""
    from providers.tokenizer import O200K_ASSET_PATH, vocabulary_is_valid

    destination = O200K_ASSET_PATH
    if destination.is_file() and vocabulary_is_valid(destination.read_bytes()):
        return

    response = client.get(_URL)
    response.raise_for_status()
    data = response.content
    if not vocabulary_is_valid(data):
        raise ValueError("Downloaded o200k_base vocabulary failed SHA-256 verification.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        staged.write_bytes(data)
        try:
            staged.replace(destination)
        except PermissionError:
            # Windows can deny a replacement while another process reads the published file.
            if not destination.is_file() or not vocabulary_is_valid(destination.read_bytes()):
                raise


if __name__ == "__main__":
    # The standalone setup command runs from a checkout without an installed application package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    with httpx.Client(timeout=30.0) as client:
        prefetch(client)

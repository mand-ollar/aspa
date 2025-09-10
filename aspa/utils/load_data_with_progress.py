from pathlib import Path

from tqdm import tqdm


def load_data_with_progress(file_path: str | Path, leave: bool = False) -> bytes:
    file_size: int = Path(file_path).stat().st_size

    with open(file=file_path, mode="rb") as f:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Loading data",
            leave=leave,
            ncols=80,
        ) as pbar:
            chunk_size: int = 8192
            data: bytes = b""

            while True:
                chunk: bytes = f.read(chunk_size)
                if not chunk:
                    break
                data += chunk
                pbar.update(len(chunk))

    return data

from typing import Iterator, Tuple, Any, Dict, List
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# ------------------------------
# Global lazy loader for TF-Hub
# ------------------------------
_EMBEDDER = None
_EMBED_DIM = 512  # USE-Large dim

def _get_embedder():
    """Load TF-Hub embedder once per process; fall back to None."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            # 인터넷이 되면 아래 라인을 쓰고,
            _EMBEDDER = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
            # 오프라인/로컬 모듈이 있으면 위를 주석, 아래처럼 경로 지정:
            # _EMBEDDER = hub.load("/path/to/local/use_large_5")
        except Exception:
            _EMBEDDER = None
    return _EMBEDDER

def _embed_text(text: str) -> np.ndarray:
    """Return 512-d float32 embedding; zeros if embedder unavailable."""
    emb = _get_embedder()
    if emb is None:
        return np.zeros((_EMBED_DIM,), dtype=np.float32)
    # emb returns tf.Tensor
    vec = emb([text])[0].numpy().astype(np.float32, copy=False)
    # 안전을 위해 차원 보정
    if vec.shape != (_EMBED_DIM,):
        out = np.zeros((_EMBED_DIM,), dtype=np.float32)
        n = min(_EMBED_DIM, vec.shape[-1])
        out[:n] = vec[:n]
        return out
    return vec


# ------------------------------
# Top-level parser for Beam
# ------------------------------
def _parse_episode_file(episode_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    episode_path: data/{train,val}/*.npy
    File contains: List[Dict] with keys:
      image(224,224,3 uint8), wrist_image(224,224,3 uint8),
      state(7 float32), action(7 float32),
      language_instruction(str), frame_index(int, optional)
    """
    data: List[Dict[str, Any]] = np.load(episode_path, allow_pickle=True).tolist()
    if not isinstance(data, list):
        raise ValueError(f"Episode is not a list: {episode_path}")

    # caption: 모든 스텝에 동일하다고 가정
    caption = ""
    for step in data:
        cap = step.get("language_instruction", "")
        if isinstance(cap, str) and cap:
            caption = cap
            break
    lang_emb = _embed_text(caption)

    steps = []
    T = len(data)
    for i, step in enumerate(data):
        # 타입/차원 안전성 보정
        img = np.asarray(step["image"], dtype=np.uint8)
        wrist = np.asarray(step["wrist_image"], dtype=np.uint8)
        if img.shape != (224, 224, 3):
            raise ValueError(f"image shape must be (224,224,3): {episode_path}")
        if wrist.shape != (224, 224, 3):
            # wrist가 없으면 zeros로 채웠을 수도 있는데, 혹시 1x1x1이면 브로드캐스트
            if wrist.size == 1:
                wrist = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                raise ValueError(f"wrist_image shape must be (224,224,3): {episode_path}")

        state = np.asarray(step.get("state", np.zeros(7, np.float32)), dtype=np.float32)
        action = np.asarray(step.get("action", np.zeros(7, np.float32)), dtype=np.float32)
        if state.shape != (7,): state = state.reshape(-1)[:7].astype(np.float32)
        if action.shape != (7,): action = action.reshape(-1)[:7].astype(np.float32)

        steps.append({
            "observation": {
                "image": img,
                "wrist_image": wrist,
                "state": state,
            },
            "action": action,
            "discount": np.float32(1.0),
            "reward": np.float32(1.0 if i == (T - 1) else 0.0),
            "is_first": bool(i == 0),
            "is_last": bool(i == (T - 1)),
            "is_terminal": bool(i == (T - 1)),
            "language_instruction": caption,
            "language_embedding": lang_emb,
        })

    sample = {
        "steps": steps,
        "episode_metadata": {"file_path": episode_path},
    }
    return episode_path, sample


# ------------------------------
# TFDS Builder
# ------------------------------
class ego4dDataset(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for Ego4D RLDS episodes saved as {take}__{action}.npy"""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata & features."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format="png",
                            doc="Main (exo) RGB."
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format="png",
                            doc="Wrist/ego RGB."
                        ),
                        "state": tfds.features.Tensor(
                            shape=(7,), dtype=np.float32,
                            doc="7D state (dummy in this dataset)."
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,), dtype=np.float32,
                        doc="7D action (dummy in this dataset)."
                    ),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                    "language_embedding": tfds.features.Tensor(
                        shape=(_EMBED_DIM,), dtype=np.float32,
                        doc="USE-Large (512D) embedding."
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define splits. Expect files under data/train and/or data/val."""
        return {
            # 필요하면 주석 해제
            # "train": self._generate_examples(path="data/train/*.npy"),
            "val": self._generate_examples(path="data/val/*.npy"),
        }

    def _generate_examples(self, path: str) -> Iterator[Tuple[str, Any]]:
        """Yield examples (with Apache Beam for parallelism)."""
        episode_paths = glob.glob(path)
        episode_paths.sort()
        beam = tfds.core.lazy_imports.apache_beam
        return (
            beam.Create(episode_paths)
            | beam.Map(_parse_episode_file)  # top-level picklable fn
        )

"""Local, explainable jewellery recommendation utilities."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


DINO_MODEL_ID = "facebook/dinov2-base"
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
CACHE_VERSION = "v1"


class InventoryError(ValueError):
    """Raised when the supplied inventory cannot be used safely."""


@dataclass(frozen=True)
class Product:
    product_id: str
    product_type: str
    image_path: Path


@dataclass(frozen=True)
class Recommendation:
    product: Product
    score: float
    explanation: str
    dino_score: float
    siglip_score: float
    palette_score: float


@dataclass(frozen=True)
class PersonalizedRanker:
    """A small regression layer learned from a user's pair-quality labels."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    label_count: int
    validation_mae: float | None

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.mean) / self.scale
        return np.clip(standardized @ self.weights + self.bias, 0.0, 3.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "label_count": self.label_count,
            "validation_mae": self.validation_mae,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PersonalizedRanker":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
            weights=np.asarray(data["weights"], dtype=np.float32),
            bias=float(data["bias"]),
            label_count=int(data["label_count"]),
            validation_mae=None if data.get("validation_mae") is None else float(data["validation_mae"]),
        )


class JewelleryRecommender:
    """Ranks catalogue earrings for a selected catalogue necklace."""

    def __init__(self, dataset_path: str | Path, image_dir: str | Path, cache_dir: str | Path):
        self.dataset_path = Path(dataset_path)
        self.image_dir = Path(image_dir)
        self.cache_dir = Path(cache_dir)
        self.products = load_inventory(self.dataset_path, self.image_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._dino_processor = None
        self._dino_model = None
        self._siglip_processor = None
        self._siglip_model = None
        self._earring_features: dict[str, dict[str, np.ndarray]] | None = None
        self.cache_was_used = False

    @property
    def necklaces(self) -> list[Product]:
        return [product for product in self.products if product.product_type == "Necklace"]

    @property
    def earrings(self) -> list[Product]:
        return [product for product in self.products if product.product_type == "Earrings"]

    @property
    def accelerator_label(self) -> str:
        if self.device.type == "cuda":
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return "CPU fallback"

    def recommend(
        self,
        necklace_id: str,
        limit: int = 3,
        personalized_ranker: PersonalizedRanker | None = None,
    ) -> list[Recommendation]:
        necklace = next((p for p in self.necklaces if p.product_id == necklace_id), None)
        if necklace is None:
            raise InventoryError(f"Unknown necklace ID: {necklace_id}")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        self._ensure_earring_features()
        query = self._features_for_image(necklace.image_path)
        ranked: list[Recommendation] = []
        component_rows: list[tuple[Product, float, float, float]] = []
        for earring in self.earrings:
            candidate = self._earring_features[earring.product_id]
            dino_score = cosine_similarity(query["dino"], candidate["dino"])
            siglip_score = cosine_similarity(query["siglip"], candidate["siglip"])
            palette_score = histogram_similarity(query["palette"], candidate["palette"])
            component_rows.append((earring, dino_score, siglip_score, palette_score))

        learned_scores = None
        if personalized_ranker is not None:
            learned_scores = personalized_ranker.predict(
                np.asarray([[dino, siglip, palette] for _, dino, siglip, palette in component_rows])
            ) / 3.0

        for index, (earring, dino_score, siglip_score, palette_score) in enumerate(component_rows):
            combined = 0.60 * dino_score + 0.30 * siglip_score + 0.10 * palette_score
            score = float(learned_scores[index]) if learned_scores is not None else float(combined)
            ranked.append(
                Recommendation(
                    product=earring,
                    score=score,
                    explanation=(
                        "Ranked by your trained compatibility model, using the three visual signals below."
                        if learned_scores is not None
                        else make_explanation(dino_score, siglip_score, palette_score)
                    ),
                    dino_score=float(dino_score),
                    siglip_score=float(siglip_score),
                    palette_score=float(palette_score),
                )
            )
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]

    def pair_features(self, necklace_id: str) -> dict[str, np.ndarray]:
        """Return the three normalized visual signals for every earring pair."""
        necklace = next((p for p in self.necklaces if p.product_id == necklace_id), None)
        if necklace is None:
            raise InventoryError(f"Unknown necklace ID: {necklace_id}")
        self._ensure_earring_features()
        query = self._features_for_image(necklace.image_path)
        return {
            earring.product_id: np.asarray(
                [
                    cosine_similarity(query["dino"], self._earring_features[earring.product_id]["dino"]),
                    cosine_similarity(query["siglip"], self._earring_features[earring.product_id]["siglip"]),
                    histogram_similarity(query["palette"], self._earring_features[earring.product_id]["palette"]),
                ],
                dtype=np.float32,
            )
            for earring in self.earrings
        }

    def _ensure_models(self) -> None:
        # A partially completed first download can leave DINO initialized while
        # SigLIP is still unavailable. Only treat model setup as complete when
        # both encoders and both processors exist.
        if all(
            component is not None
            for component in (
                self._dino_processor,
                self._dino_model,
                self._siglip_processor,
                self._siglip_model,
            )
        ):
            return
        if self._dino_model is None:
            self._dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
            self._dino_model = AutoModel.from_pretrained(DINO_MODEL_ID, dtype=self.dtype).to(self.device).eval()
        if self._siglip_model is None:
            self._siglip_processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_ID)
            self._siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, dtype=self.dtype).to(self.device).eval()

    def _features_for_image(self, image_path: Path) -> dict[str, np.ndarray]:
        self._ensure_models()
        image = Image.open(image_path).convert("RGB")
        with torch.inference_mode():
            dino_inputs = self._dino_processor(images=image, return_tensors="pt")
            dino_pixels = dino_inputs["pixel_values"].to(self.device, dtype=self.dtype)
            dino_output = self._dino_model(pixel_values=dino_pixels)
            dino_vector = dino_output.last_hidden_state[:, 0, :].float().cpu().numpy()[0]

            siglip_inputs = self._siglip_processor(images=image, return_tensors="pt")
            siglip_pixels = siglip_inputs["pixel_values"].to(self.device, dtype=self.dtype)
            siglip_output = self._siglip_model.get_image_features(pixel_values=siglip_pixels)
            # Transformers exposes SigLIP2 image features differently across
            # its multimodal and vision-only model classes.
            if isinstance(siglip_output, torch.Tensor):
                siglip_vector = siglip_output
            elif getattr(siglip_output, "image_embeds", None) is not None:
                siglip_vector = siglip_output.image_embeds
            elif getattr(siglip_output, "pooler_output", None) is not None:
                siglip_vector = siglip_output.pooler_output
            else:
                raise RuntimeError("SigLIP2 did not return an image embedding.")
            siglip_vector = siglip_vector.float().cpu().numpy()[0]

        return {
            "dino": l2_normalize(dino_vector),
            "siglip": l2_normalize(siglip_vector),
            "palette": extract_palette_features(image),
        }

    def _ensure_earring_features(self) -> None:
        if self._earring_features is not None:
            return
        cache_path = self.cache_dir / f"earring_features_{CACHE_VERSION}.npz"
        signature = inventory_signature(self.earrings, self.dataset_path)
        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if str(cached["signature"][0]) == signature:
                    ids = cached["ids"].tolist()
                    self._earring_features = {
                        product_id: {
                            "dino": cached["dino"][index],
                            "siglip": cached["siglip"][index],
                            "palette": cached["palette"][index],
                        }
                        for index, product_id in enumerate(ids)
                    }
                    self.cache_was_used = True
                    return
            except (OSError, KeyError, ValueError):
                pass

        features = {earring.product_id: self._features_for_image(earring.image_path) for earring in self.earrings}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            signature=np.array([signature]),
            ids=np.array(list(features)),
            dino=np.stack([features[product_id]["dino"] for product_id in features]),
            siglip=np.stack([features[product_id]["siglip"] for product_id in features]),
            palette=np.stack([features[product_id]["palette"] for product_id in features]),
        )
        self._earring_features = features


def load_inventory(dataset_path: Path, image_dir: Path) -> list[Product]:
    """Load the provided CSV and ensure its records are usable."""
    if not dataset_path.is_file():
        raise InventoryError(f"Dataset was not found: {dataset_path}")
    if not image_dir.is_dir():
        raise InventoryError(f"Image folder was not found: {image_dir}")

    required = {"id", "product_type", "image_file"}
    products: list[Product] = []
    seen_ids: set[str] = set()
    with dataset_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise InventoryError("CSV must contain id, product_type, and image_file columns.")
        for row_number, row in enumerate(reader, start=2):
            product_id = (row.get("id") or "").strip()
            product_type = (row.get("product_type") or "").strip()
            image_file = (row.get("image_file") or "").strip()
            if not product_id or product_id in seen_ids:
                raise InventoryError(f"Row {row_number} has a missing or duplicate id.")
            if product_type not in {"Necklace", "Earrings"}:
                raise InventoryError(f"Row {row_number} has unsupported product_type: {product_type!r}")
            image_path = image_dir / image_file
            if not image_path.is_file():
                raise InventoryError(f"Image listed on row {row_number} was not found: {image_file}")
            seen_ids.add(product_id)
            products.append(Product(product_id, product_type, image_path))

    if len([p for p in products if p.product_type == "Necklace"]) == 0:
        raise InventoryError("Inventory contains no necklaces.")
    if len([p for p in products if p.product_type == "Earrings"]) < 3:
        raise InventoryError("Inventory needs at least three earrings.")
    return products


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def extract_palette_features(image: Image.Image) -> np.ndarray:
    """Return a stable Lab/HSV histogram while discounting a uniform studio background."""
    rgb = np.asarray(image.resize((256, 256))).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    border = np.concatenate((rgb[:12, :, :].reshape(-1, 3), rgb[-12:, :, :].reshape(-1, 3),
                             rgb[:, :12, :].reshape(-1, 3), rgb[:, -12:, :].reshape(-1, 3)))
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    foreground = distance > 18.0
    if foreground.sum() < 250:
        foreground = np.ones(rgb.shape[:2], dtype=bool)

    lab_hist = cv2.calcHist([lab], [1, 2], foreground.astype(np.uint8), [8, 8], [0, 256, 0, 256]).flatten()
    hsv_hist = cv2.calcHist([hsv], [0, 1], foreground.astype(np.uint8), [8, 8], [0, 180, 0, 256]).flatten()
    return l2_normalize(np.concatenate((lab_hist, hsv_hist)).astype(np.float32))


def histogram_similarity(left: np.ndarray, right: np.ndarray) -> float:
    # Palette vectors are non-negative and L2-normalized, so cosine similarity is in [0, 1].
    return float(np.clip(np.dot(left, right), 0.0, 1.0))


def make_explanation(dino_score: float, siglip_score: float, palette_score: float) -> str:
    components = {
        "fine visual motif and texture": dino_score,
        "overall jewellery style": siglip_score,
        "metal tone and colour palette": palette_score,
    }
    ordered = sorted(components, key=components.get, reverse=True)
    strongest, next_strongest = ordered[:2]
    return f"Best alignment on {strongest}, supported by {next_strongest}."


def inventory_signature(products: Iterable[Product], dataset_path: Path) -> str:
    """Invalidate cached features when the inventory CSV or its images change."""
    digest = hashlib.sha256()
    digest.update(dataset_path.read_bytes())
    for product in sorted(products, key=lambda item: item.product_id):
        stat = product.image_path.stat()
        digest.update(f"{product.product_id}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def load_pair_labels(path: str | Path) -> dict[tuple[str, str], int]:
    """Read manually supplied 0–3 compatibility labels, if present."""
    label_path = Path(path)
    if not label_path.exists():
        return {}
    labels: dict[tuple[str, str], int] = {}
    with label_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                value = int(row["quality"])
                if value not in {0, 1, 2, 3}:
                    continue
                labels[(row["necklace_id"], row["earring_id"])] = value
            except (KeyError, TypeError, ValueError):
                continue
    return labels


def save_pair_labels(path: str | Path, labels: dict[tuple[str, str], int]) -> None:
    label_path = Path(path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["necklace_id", "earring_id", "quality"])
        writer.writeheader()
        for (necklace_id, earring_id), quality in sorted(labels.items()):
            writer.writerow({"necklace_id": necklace_id, "earring_id": earring_id, "quality": quality})


def fit_personalized_ranker(features: np.ndarray, targets: np.ndarray, seed: int = 42) -> PersonalizedRanker:
    """Fit a regularized linear ranker with a held-out MAE when data permits."""
    if features.ndim != 2 or features.shape[1] != 3 or len(features) != len(targets):
        raise ValueError("features must be an N×3 matrix paired with N targets")
    if len(features) < 12:
        raise ValueError("At least 12 labelled pairs are needed before training.")
    if np.any((targets < 0) | (targets > 3)):
        raise ValueError("Targets must be manual compatibility scores from 0 to 3.")

    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(features))
    validation_size = max(2, round(len(features) * 0.2)) if len(features) >= 15 else 0
    validation_indices, training_indices = indices[:validation_size], indices[validation_size:]
    train_features, train_targets = features[training_indices], targets[training_indices]
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (train_features - mean) / scale

    # Ridge regression makes the small-label regime stable and fully reproducible.
    design = np.column_stack((standardized, np.ones(len(standardized))))
    regularizer = np.diag([0.15, 0.15, 0.15, 0.0])
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ train_targets)
    ranker = PersonalizedRanker(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        weights=coefficients[:3].astype(np.float32),
        bias=float(coefficients[3]),
        label_count=len(features),
        validation_mae=None,
    )
    if validation_size:
        mae = np.abs(ranker.predict(features[validation_indices]) - targets[validation_indices]).mean()
        ranker = PersonalizedRanker(**{**ranker.__dict__, "validation_mae": float(mae)})
    return ranker


def save_personalized_ranker(path: str | Path, ranker: PersonalizedRanker) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(ranker.to_dict(), indent=2), encoding="utf-8")


def load_personalized_ranker(path: str | Path) -> PersonalizedRanker | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    try:
        return PersonalizedRanker.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None

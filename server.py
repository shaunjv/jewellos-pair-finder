"""FastAPI server for the local Jewellos visual matching prototype."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recommender import (
    InventoryError,
    JewelleryRecommender,
    fit_personalized_ranker,
    load_pair_labels,
    load_personalized_ranker,
    save_pair_labels,
    save_personalized_ranker,
)


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "ASSIGNMENT" / "candidate_dataset.csv"
IMAGE_DIR = ROOT / "ASSIGNMENT" / "Jewelry Images"
CACHE_DIR = ROOT / "cache"
LABELS_PATH = ROOT / "labels" / "pair_labels.csv"
RANKER_PATH = CACHE_DIR / "personalized_ranker.json"

app = FastAPI(title="Jewellos Pair Finder")
app.mount("/assets", StaticFiles(directory=ROOT / "static"), name="assets")
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
_recommender: JewelleryRecommender | None = None
_model_lock = Lock()


class RatingsPayload(BaseModel):
    necklace_id: str
    ratings: dict[str, int]


def get_recommender() -> JewelleryRecommender:
    global _recommender
    # This creates the main matching object only once.
    # It reads the CSV and knows all necklaces and earrings.
    # Keeping it in memory means we do not start from zero for every click.
    if _recommender is None:
        _recommender = JewelleryRecommender(DATASET, IMAGE_DIR, CACHE_DIR)
    return _recommender


def product_payload(product) -> dict[str, str]:
    return {"id": product.product_id, "type": product.product_type, "image": f"/images/{product.image_path.name}"}


def recommendation_payload(item) -> dict[str, object]:
    return {
        "id": item.product.product_id,
        "image": f"/images/{item.product.image_path.name}",
        "score": max(0, min(100, round(item.score * 100))),
        "explanation": item.explanation,
        "details": {"fine_detail": round(item.dino_score, 3), "style": round(item.siglip_score, 3), "palette": round(item.palette_score, 3)},
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/products")
def products() -> dict[str, object]:
    # app.js calls this when the page first opens.
    # We send the list of necklaces and earrings back to the browser.
    # The browser uses it to fill the dropdown boxes.
    catalog = get_recommender()
    return {"necklaces": [product_payload(item) for item in catalog.necklaces], "earrings": [product_payload(item) for item in catalog.earrings]}


@app.get("/api/recommendations/{necklace_id}")
def recommendations(necklace_id: str) -> dict[str, object]:
    # app.js sends the selected necklace ID here.
    # Example URL: /api/recommendations/N01
    # This function asks recommender.py to rank all earrings for that necklace.
    catalog = get_recommender()
    try:
        # One model job runs at a time.
        # This prevents two browser clicks from trying to use the GPU together.
        with _model_lock:
            ranked = catalog.recommend(necklace_id, limit=len(catalog.earrings), personalized_ranker=load_personalized_ranker(RANKER_PATH))
    except InventoryError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    payload = [recommendation_payload(item) for item in ranked]
    return {"top": payload[:3], "remaining": payload[3:], "cached": catalog.cache_was_used}


@app.get("/api/labels")
def labels() -> dict[str, object]:
    saved = load_pair_labels(LABELS_PATH)
    return {"count": len(saved), "ratings": [{"necklace_id": necklace, "earring_id": earring, "quality": quality} for (necklace, earring), quality in saved.items()]}


@app.post("/api/labels")
def save_labels(payload: RatingsPayload) -> dict[str, int]:
    # The browser sends ratings from 0 to 3 here.
    # We check that the necklace, earrings, and rating values are valid.
    # Then we save the ratings in a local CSV file.
    # This does not train anything yet.
    catalog = get_recommender()
    valid_necklaces = {item.product_id for item in catalog.necklaces}
    valid_earrings = {item.product_id for item in catalog.earrings}
    if payload.necklace_id not in valid_necklaces:
        raise HTTPException(status_code=400, detail="Unknown necklace.")
    if any(earring not in valid_earrings or rating not in {0, 1, 2, 3} for earring, rating in payload.ratings.items()):
        raise HTTPException(status_code=400, detail="Ratings must use valid earrings and values from 0 to 3.")
    saved = load_pair_labels(LABELS_PATH)
    saved.update({(payload.necklace_id, earring): rating for earring, rating in payload.ratings.items()})
    save_pair_labels(LABELS_PATH, saved)
    return {"count": len(saved)}


@app.post("/api/train")
def train() -> dict[str, object]:
    # This starts after the user clicks Train.
    # We read all saved ratings from the CSV file.
    # For every rating, we get its three image-match scores.
    # Then we train a small model using those scores and ratings.
    saved = load_pair_labels(LABELS_PATH)
    if len(saved) < 12:
        raise HTTPException(status_code=400, detail=f"Add {12 - len(saved)} more ratings before training.")
    catalog = get_recommender()
    feature_rows, targets = [], []
    with _model_lock:
        for necklace_id in sorted({necklace for necklace, _ in saved}):
            features = catalog.pair_features(necklace_id)
            for (necklace, earring), quality in saved.items():
                if necklace == necklace_id:
                    feature_rows.append(features[earring])
                    targets.append(quality)
        import numpy as np
        ranker = fit_personalized_ranker(np.asarray(feature_rows), np.asarray(targets, dtype=np.float32))
        # Save the trained model in a small JSON file.
        # Next time the user asks for matches, this file is used.
        # So the results can follow the user's own ratings.
        save_personalized_ranker(RANKER_PATH, ranker)
    return {"label_count": ranker.label_count, "validation_mae": ranker.validation_mae}

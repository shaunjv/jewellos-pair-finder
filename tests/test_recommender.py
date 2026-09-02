from pathlib import Path

import numpy as np

from recommender import (
    JewelleryRecommender,
    cosine_similarity,
    fit_personalized_ranker,
    histogram_similarity,
    inventory_signature,
    load_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "ASSIGNMENT" / "candidate_dataset.csv"
IMAGE_DIR = ROOT / "ASSIGNMENT" / "Jewelry Images"


def test_supplied_inventory_has_expected_valid_products():
    products = load_inventory(DATASET, IMAGE_DIR)
    assert len(products) == 20
    assert len([product for product in products if product.product_type == "Necklace"]) == 5
    assert len([product for product in products if product.product_type == "Earrings"]) == 15
    assert all(product.image_path.is_file() for product in products)


def test_similarity_helpers_are_bounded():
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])
    assert cosine_similarity(first, first) == 1.0
    assert cosine_similarity(first, second) == 0.0
    assert 0.0 <= histogram_similarity(first, second) <= 1.0


def test_inventory_signature_is_stable_for_unchanged_inputs():
    products = load_inventory(DATASET, IMAGE_DIR)
    earrings = [product for product in products if product.product_type == "Earrings"]
    assert inventory_signature(earrings, DATASET) == inventory_signature(earrings, DATASET)


def test_recommender_exposes_only_inventory_earrings_without_loading_models(tmp_path):
    recommender = JewelleryRecommender(DATASET, IMAGE_DIR, tmp_path)
    assert len(recommender.necklaces) == 5
    assert len(recommender.earrings) == 15
    assert all(product.product_type == "Earrings" for product in recommender.earrings)


def test_small_personalized_ranker_learns_from_three_visual_features():
    features = np.array([[i / 20, i / 25, i / 30] for i in range(20)], dtype=np.float32)
    targets = np.clip(features[:, 0] * 3, 0, 3)
    ranker = fit_personalized_ranker(features, targets)
    predictions = ranker.predict(features)
    assert ranker.label_count == 20
    assert ranker.validation_mae is not None
    assert predictions[-1] > predictions[0]

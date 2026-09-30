import joblib
from hybrid_model import HybridClassifierPipeline, extract_linguistic_features

# Load the saved model
pipe = joblib.load('models/hybrid_feature_classifier.joblib')

test_samples = [
    ("Aspirational policy", "We are committed to climate action and support the UN SDGs as part of our long term vision."),
    ("Substantive metric", "In FY2023, our solar installations reduced Scope 1 GHG emissions by 28.5% across 14 facilities, verified by Bureau Veritas."),
    ("General compliance", "Drillisch AG strictly adheres to its compliance directive and general policy principles."),
    ("Substantive CaPEx", "We invested €14.2 million in building energy efficiency and heat pumps, lowering electricity consumption by 19%."),
]

passages = [p for _, p in test_samples]
preds = pipe.predict(passages)
probs = pipe.predict_proba(passages)

print("\n" + "=" * 80)
print("  SAVED HYBRID CLASSIFIER INFERENCE VERIFICATION")
print("=" * 80)
for (label, text), pred, prob in zip(test_samples, preds, probs):
    print(f"  [{pred.upper()}] (Substantive Prob: {prob*100:5.1f}%) | Category: {label}")
    print(f"    Text: \"{text}\"")
print("=" * 80 + "\n")

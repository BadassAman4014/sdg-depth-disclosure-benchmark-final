import joblib

pipe = joblib.load('models/hybrid_feature_classifier.joblib')
print("Model classes:", pipe.classes_)
test_sentences = [
    "We are committed to climate action and support the UN SDGs as part of our long term vision.",
    "In FY2023, we reduced Scope 1 GHG emissions by 28.5% across 14 facilities, verified by Bureau Veritas.",
    "Drillisch AG strictly adheres to its compliance directive and code of conduct."
]
raw_probs = pipe.clf.predict_proba(pipe.transform_features(test_sentences))
print("Raw predict_proba matrix:\n", raw_probs)
print("Classes:", pipe.clf.classes_)

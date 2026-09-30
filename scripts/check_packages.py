import sys

for pkg in ['interpret', 'shap', 'lightgbm', 'xgboost']:
    try:
        m = __import__(pkg)
        print(f"{pkg}: INSTALLED (version {getattr(m, '__version__', 'unknown')})")
    except ImportError:
        print(f"{pkg}: NOT installed")

try:
    import numpy as np
    print("NumPy imported successfully!")
except Exception as e:
    print(f"NumPy import failed: {e}")

try:
    import pandas as pd
    print("Pandas imported successfully!")
except Exception as e:
    print(f"Pandas import failed: {e}")

try:
    import sklearn
    print("scikit-learn basic import successfully!")
except Exception as e:
    print(f"scikit-learn basic import failed: {e}")

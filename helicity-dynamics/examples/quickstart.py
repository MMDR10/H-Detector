"""
Example: Quickstart — 5 lines to classify any time series.

This is the simplest possible usage demonstration.
"""

import numpy as np
import helicity_dynamics as hd

# 1. Create or load your data (n_steps, n_features)
data = np.random.RandomState(42).randn(200, 5)

# 2. Compute Ô
curl, helicity, balance = hd.compute(data)

# 3. See the numbers
print(f"curl={curl:.4f}  helicity={helicity:.4f}  balance={balance:.4f}")

# 4. Classify
print(hd.classify(helicity, balance))

# 5. Full spectrum report
import json
print(json.dumps(hd.spectrum_classify(helicity, balance, curl), indent=2))

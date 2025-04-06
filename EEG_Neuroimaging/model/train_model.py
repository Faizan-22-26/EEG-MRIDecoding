import numpy as np
from model import build_model

X = np.random.rand(100, 128, 1)
y = np.random.rand(100, 1)

model = build_model((128, 1))
model.fit(X, y, epochs=10, batch_size=16)
model.save("model/saved_model/")

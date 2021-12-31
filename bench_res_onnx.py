from time import perf_counter

import numpy as np
import onnxruntime as rt

dim = 320
model = rt.InferenceSession("networks/NFNetL1V1-100-0.57141.onnx")

img = np.random.rand(1, dim, dim, 3).astype(np.float32)

input_name = model.get_inputs()[0].name
label_name = model.get_outputs()[0].name
probs = model.run([label_name], {input_name: img})[0]

runs = 100
times = []
for _ in range(runs):
    start = perf_counter()
    _ = model.run([label_name], {input_name: img})
    times.append(perf_counter() - start)
print((sum(times) * 1000) / runs)

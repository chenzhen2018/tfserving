# -*- coding: utf-8 -*-  
# =================================================

import numpy as np
from PIL import Image
from libs.deploy import PredictModelGrpc, PredictModelRESTAPI

"""
4. Client
"""

# =======================
# ===== Load image ======
# =======================
image_path = './test_images/img_1.png'
img = Image.open(image_path)
img = np.array(img) / 255.0

# =======================
# ====== Predict ========
# =======================
model = PredictModelRESTAPI(model_name='clothing', input_name='flatten_input', output_name='dense_1', socket='localhost:8501')
res = model.inference(img)
print(res)

model = PredictModelGrpc(model_name='clothing', input_name='flatten_input', output_name='dense_1', socket='0.0.0.0:8500')
res = model.inference(img)
print(res)



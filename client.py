# -*- coding: utf-8 -*-  
# =================================================

import numpy as np
from PIL import Image
from deploy import PredictModelGrpc

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
# ===== Load image ======
# =======================
model = PredictModelGrpc(model_name='clothing', input_name='flatten_input', output_name='dense_1')
res = model.inference(img)
print(res)



import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3, 7"
from main.model.pnet import VideoCompressor as Model
from thop import profile
from thop import clever_format
import torch

model = Model().cuda()

flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 4, 3, 256, 256).cuda()))
print(flops)
print(params)
flops, params = clever_format([flops, params], "%.3f")
print(flops)
print(params)

# all
# 247479215692.0
# 26238325.0
# 247.479G
# 26.238M

# wo_offset
# 231073719240.0
# 25668721.0
# 231.074G
# 25.669M

# wo_mcfilter
# 199190193096.0
# 26022577.0
# 199.190G
# 26.023M

# wo_fix
# 201375425480.0
# 25534449.0
# 201.375G
# 25.534M
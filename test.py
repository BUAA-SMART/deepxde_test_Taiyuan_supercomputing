import torch
import time

print(torch.__version__)

a = torch.randn(10000,10000)
b = torch.randn(10000,10000)


s = time.time()
c = a.matmul(b)
print("CPU time:", time.time() - s)

a = a.cuda()
b = b.cuda()

s = time.time()
c = a.matmul(b)
print("DCU time:", time.time() - s)

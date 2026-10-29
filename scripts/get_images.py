import os
import urllib.request

os.makedirs("my_images", exist_ok=True)
ids = [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761]
ok = 0
for i in ids:
    path = f"my_images/coco_{i:012d}.jpg"
    if os.path.exists(path):
        ok += 1
        continue
    try:
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{i:012d}.jpg", path)
        ok += 1
    except Exception as e:
        print("skip", i, e)
print(ok, "images in my_images")
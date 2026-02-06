import lmdb
import matplotlib.pyplot as plt
import cv2
import pickle

env1 = lmdb.open("/home/wjm/MyFinalProject/test-ll-extreme-crop.lmdb", subdir=False, readonly=True)
ext = env1.begin()
env2 = lmdb.open("/home/wjm/MyFinalProject/test-ll-hard-crop.lmdb", subdir=False, readonly=True)
hard = env2.begin()
env3 = lmdb.open("/home/wjm/MyFinalProject/test-ll-normal-crop.lmdb", subdir=False, readonly=True)
norm = env3.begin()

def show(db,num):
    img,_ = pickle.loads(db.get(str(num).encode("ascii")))
    img = img.squeeze(0)
    img = img.permute([1,2,0])
    return img


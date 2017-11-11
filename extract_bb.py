import numpy as np
np.set_printoptions(threshold=np.nan)

X_train = np.load('./data/train_X.npy')
y_train = np.load('./data/train_Y.npy')
bboxes  = np.load('./data/train_bboxes.npy')

box1 = bboxes[0]
print("number 1:")
print("top left: [" + str(box1[0][1]) + ", " + str(box1[0][0]) + "]") 
print("bottom right: [" + str(box1[0][3]) + ", " + str(box1[0][2]) + "]") 

print("number 2:")
print("top left: [" + str(box1[1][1]) + ", " + str(box1[1][0]) + "]") 
print("bottom right: [" + str(box1[1][3]) + ", " + str(box1[1][2]) + "]") 

# 64 * upper left y + upper left x = starting value of bounding box
img1  = np.array([])
start = box1[0][0] * 64 + box1[0][1]
end   = start + 28
for i in range(28):
	img1 = np.append(img1, X_train[0][start:end])
	start += 64
	end += 64

print(img1.reshape([28,28]))
import matplotlib.pyplot as plt
import cv2


original_image_path = r"C:\Users\minju\Desktop\code\unet_tiktok\data\images\108_00270.png"
image_path1 = r"C:\Users\minju\Desktop\code\unet_tiktok\data\masks\108_00270.png"
image_path2 = r"C:\Users\minju\Desktop\code\unet_tiktok\new_results\108_00270.png"

img = cv2.imread(original_image_path)
img1 = cv2.imread(image_path1)
img2 = cv2.imread(image_path2)
#print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Visualization
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title("Input Image")
axes[0].axis("off")  

axes[1].imshow(img1)
axes[1].set_title("Ground Truth")
axes[1].axis("off")  

axes[2].imshow(img2)
axes[2].set_title("Prediction")
axes[2].axis("off")  

plt.tight_layout()
plt.show()

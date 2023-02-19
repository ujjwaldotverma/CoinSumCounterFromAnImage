import cv2

# Load the image
img = cv2.imread('Five2.png')

# Define the new resolution
width = 300
height = 300

# Resize the image to the new resolution
resized_img = cv2.resize(img, (width, height))

# Save the resized image
cv2.imwrite('resized_image.jpg', resized_img)

import cv2

# original image
print('original image')
img1 = cv2.imread(r"D:\Image_Processing and Computer Vision\avenger.jpeg", 1)

if img1 is None:
    print("Image not found")
else:
    img1 = cv2.resize(img1, (1280, 700))  # width, height
    print(img1)
    cv2.imshow("original", img1)



# convert colored into grayscale
print('grayscale image')
img2 = cv2.imread(r"D:\Image_Processing and Computer Vision\avenger.jpeg", 0)

if img2 is None:
    print("Image not found")
else:
    img2 = cv2.resize(img2, (1280, 700))
    print(img2)
    cv2.imshow("grayscale image", img2)




# read image as-is (NOT saturation, just your flag usage)
print('as-is image')
img3 = cv2.imread(r"D:\Image_Processing and Computer Vision\avenger.jpeg", -1)

if img3 is None:
    print("Image not found")
else:
    img3 = cv2.resize(img3, (1280, 700))
    print(img3)
    cv2.imshow("as-is image", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

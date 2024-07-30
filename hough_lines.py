import cv2
import numpy as np
import matplotlib.pyplot as plt
file1=cv2.imread("CFD_005.jpg")
#cv2.imshow("image1", file1)
file2=cv2.imread("cracktree200_6303.jpg")

file3=cv2.imread("DeepCrack_11166-1.jpg")
#cv2.imshow("image3", file3)

file=file3
cv2.imshow("image", file)
def HoughLines(img):
    edges = cv2.Canny(img, 50, 155)
    #cv2.imshow("edges", edges)

    lines=cv2.HoughLines(edges, 1, np.pi/180, 30)
    print(lines.shape)
    for line in lines:
        rho, theta=line[0]
        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        x1=int(x0+1000*(-b))
        x2=int(x0-1000*(-b))
        y1 = int(y0 + 1000 * (a))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)

def HoughLinesP(img):
    edges = cv2.Canny(img, 50, 155)
    # cv2.imshow("edges", edges)
    img1=img.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20)
    num_lines=lines.shape[0]
    angles=[]
    for line in lines:
        x1,y1,x2,y2=line[0]
        if not (x1, y1)==(x2,y2):
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            a=x2-x1
            b=y2-y1
            angle = np.arctan2(b, a)
            x0,y0=x1,y1
            x1 = int(x0 + 15 * (a))
            x2 = int(x0 - 15* (a))
            y1 = int(y0 + 15 * (b))
            y2 = int(y0 - 15* (b))

            angles.append(angle)
            cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return num_lines, angles, img1

num_lines, angles, img=HoughLinesP(file)
print("num_lines=", num_lines)

cv2.imshow("image1", file)
cv2.imshow("image2", img)
k=cv2.waitKey(0)

plt.hist(angles, bins=10, color='b', alpha=0.7)
plt.xlabel('Angle')
plt.ylabel('Frequency')
plt.title('Distribution of Line Angles')
plt.show()

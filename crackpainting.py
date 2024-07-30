import cv2
import numpy as np
import matplotlib.pyplot as plt


width = height = 400

center_x = center_y = width // 2

def prepare_points(x, y):
    points = []
    for c in zip(x, y):
        points.append(c)

    points = np.array(points, dtype=np.int32)
    return points

def draw_sine(image, length, amplitude, noice_level, angle, T, x_start, y_start, rad):
    num_points=50
    noice_points=int(0.2*num_points)
    noice_indices = np.random.choice(range(num_points), size=noice_points, replace=False)
    noise = np.random.normal(0, 0.5, num_points)
    noise[noice_indices] = np.random.normal(0, noice_level, noice_points)
    x = np.linspace(0, length, num_points)+rad
    y = amplitude * np.sin((x-rad) * np.pi * 2 / T) + noise
    x, y=rotate(x, y, angle)
    x+=x_start
    y+= amplitude / 2+y_start

    points=prepare_points(x, y)
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img, [points], isClosed=False, color=(255, 255, 255), thickness=2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def rotate(x, y, angle):
    com=np.array([y, x])
    rotate_matrix=np.array([[np.cos(angle*np.pi/180), -np.sin(angle*np.pi/180)],
                           [np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180)]])
    res=rotate_matrix@com
    return res[1], res[0]


def draw_circle(image, radius, noice_level, x_center, y_center, num_points):

    noice_points = int(0.4 * num_points)
    noice_indices1 = np.random.choice(range(num_points), size=noice_points, replace=False)
    noice_indices2 = np.random.choice(range(num_points), size=noice_points, replace=False)
    noise1 = np.random.normal(0, 0.5, num_points)
    noise2 = np.random.normal(0, 0.5, num_points)
    noise1[noice_indices1] = np.random.normal(0, noice_level, noice_points)
    noise2[noice_indices2] = np.random.normal(0, noice_level, noice_points)
    rad1=radius+noise1
    rad2=radius+noise2
    x1 = np.linspace(-radius, radius, num_points)
    x2= np.linspace(radius, -radius, num_points)
    y1= np.sqrt(np.abs(rad1**2-x1**2))
    y2= -np.sqrt(np.abs(rad2**2-x2**2))
    x=np.concatenate((x1, x2))+x_center
    y=np.concatenate((y1, y2))+y_center
    points = prepare_points(x, y)

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img, [points], isClosed=True, color=(255, 255, 255), thickness=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def draw_curicen(image, radius, noice_level, amplitude, number, x_start, y_start, rad):
    angles=np.linspace(0, 360, number)
    img=image

    for angle in angles:
        T=np.random.randint(radius//4, radius, 1)
        img=draw_sine(img, radius, amplitude, noice_level, angle, T, x_start, y_start, rad)

    return img


image = np.zeros((height, width)).astype(np.uint8)




img=draw_sine(image, 450, 5, 2, 359, 150, 0, 220, 0)
img=draw_sine(img, 50, 3, 1, 80, 200, 210, 220, 0)
img=draw_sine(img, 100, 5, 2, 350, 100, 180, 150, 0)
img=draw_sine(img, 50, 5, 2, 275, 65, 275, 166, 0)
img=draw_sine(img, 20, 3, 1, 180, 40, 177, 152, 0)
img=draw_sine(img, 20, 3, 1, 90, 40, 280, 170, 0)
img = draw_curicen(img, 30, 3, 10, 7, 50, 50, 25)
img = draw_circle(img, 25, 2, 50, 50, 50)
plt.imshow(img, cmap='gray')
plt.show()

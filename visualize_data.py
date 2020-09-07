import pandas
import cv2
import numpy as np
import matplotlib.pyplot as plt

# globals
columns = 200
rows = 66
channels = 3
crop_top = 50
crop_bottom = 25

def convert_grayscale(image, steering):
    gray = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)
    # convert it back to rgb
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray, steering

def translate_image(image, steering):
    range = 10 #max 10px movement
    trans_x = range*np.random.uniform()-range/2
    trans_y = range*np.random.uniform()-range/2
    TranslationMatrix = np.float32([[1,0,trans_x],[0,1,trans_y]])
    
    translated = cv2.warpAffine(image,TranslationMatrix,(image.shape[1],image.shape[0]))
    
    # recalculate steering angle for horizontal 
    steering += trans_x * 3.75e-4
    return translated, steering

def flip_image(image, steering):
    image = cv2.flip(np.copy(image),1)
    steering = -steering
    return image, steering

def add_brightness(image, steering):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = 0.8 + 0.4*(2*np.random.uniform()-1.0)
    cv2.add(hsv[:,:,2], value, hsv[:,:,2])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image, steering

def no_augmentation(image, steering):
    return image, steering

augmentation = [no_augmentation, convert_grayscale, translate_image, flip_image, add_brightness]

def get_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # remove sky and hood of the car
    image = image[35:135,:,:] # 35px from top 25 px from bottom
    
    # resize to nvidias neural network
    image = cv2.resize(image, (columns, rows), interpolation = cv2.INTER_AREA)
    
    return np.array(image)


path = "data/"
csv_file = path + "driving_log.csv"
#center,left,right,steering,throttle,brake,speed
data = pandas.read_csv(csv_file, header= 0, names = ['center', 'left', 'right', 'steering', 'throttle','brake','speed'])

image_paths = []
steering_angles = []

for row in range(int(len(data))):
    row_data = data.iloc[[row]].reset_index()

    image_paths.append(path+row_data['center'][0].strip())
    image_paths.append(path+row_data['left'][0].strip())
    image_paths.append(path+row_data['right'][0].strip())
    steering_angles.append(row_data['steering'][0])
    steering_angles.append(row_data['steering'][0] + 0.25)
    steering_angles.append(row_data['steering'][0] - 0.25)
   
image_paths = np.array(image_paths)
steering_angles = np.array(steering_angles)

plt.figure()

hist, classes = np.histogram(steering_angles, 23)
width = 0.7 * (classes[1] - classes[0])
center = (classes[:-1] + classes[1:]) / 2
plt.bar(center, hist, align='center', width = width)
plt.plot((np.min(steering_angles), np.max(steering_angles)), (len(steering_angles) / 23, len(steering_angles) / 23), 'k-')
plt.savefig('images/histo_train.png')

quit()

image_center = get_image(image_paths[30])
image_left = get_image(image_paths[31])
image_right = get_image(image_paths[32])

#cv2.imwrite("images/image_left.png", cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB))
#cv2.imwrite("images/image_center.png", cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB))
#cv2.imwrite("images/image_right.png", cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB))

image_gray, steering = convert_grayscale(image_center, steering_angles[30])
cv2.imwrite("images/image_gray.png", cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB))

image_translate, steering = translate_image(image_center, steering_angles[30])
cv2.imwrite("images/image_translate.png", cv2.cvtColor(image_translate, cv2.COLOR_BGR2RGB))

image_flip, steering = flip_image(image_center, steering_angles[30])
cv2.imwrite("images/image_flip.png", cv2.cvtColor(image_flip, cv2.COLOR_BGR2RGB))

image_bright, steering = add_brightness(image_center, steering_angles[30])
cv2.imwrite("images/image_bright.png", cv2.cvtColor(image_bright, cv2.COLOR_BGR2RGB))




# load data
#csv_file = "/opt/testdata/normal_lap/driving_log.csv"
#center,left,right,steering,throttle,brake,speed
#data = pandas.read_csv(csv_file, header= 0, names = ['center', 'left', 'right', 'steering', 'throttle','brake','speed'])

#print(len(data))
# load data
#csv_file = "data/driving_log.csv"
#center,left,right,steering,throttle,brake,speed
#data = pandas.read_csv(csv_file, header= 0, names = ['center', 'left', 'right', 'steering', 'throttle','brake','speed'])
#print(len(data))


# load data
#csv_file = "/opt/testdata/reentering/driving_log.csv"
#center,left,right,steering,throttle,brake,speed
#data = pandas.read_csv(csv_file, header= 0, names = ['center', 'left', 'right', 'steering', 'throttle','brake','speed'])
#print(len(data))
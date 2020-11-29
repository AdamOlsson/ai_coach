from torchvision.io import read_video
from torchvision.utils import save_image
import os, shutil

import matplotlib.pyplot as plt

root                = "exercises"
path_data           = "../../data/videos/{}/".format(root) # path relative this file
path_image_save_log = "../../data/images/{}/".format(root)
path_csv_save_loc   = "../../data/images/annotations.csv"  # path relative this file
exclude = ["clean_and_jerk", "power_clean_and_jerk", "push_press_and_jerk", "snatch_balance",
            "jerk", "power_snatch_and_snatch", "snatch_and_power_snatch", "frontsquat", "power_snatch", "snatch_pull"]

# Debugging
def show_frame(frame):
    fig, ax = plt.subplots(1,1, figsize=(15,8))
    ax.imshow(frame.numpy())
    plt.show()



# remove all images
print("### DELETING OLD IMAGES")
try:
    shutil.rmtree(path_image_save_log)
except OSError:
    print(" ### Directory '{}' does not exist, no delete performed.".format(path_image_save_log))

os.makedirs(path_image_save_log)

with open(path_csv_save_loc,'w+') as f:
    data = f.read()
    f.seek(0)
    f.write("# filename,label\n") # Header
    for dir in os.listdir(path_data):
        
        if dir in exclude:
            print("### SKIPPING {}".format(dir))
            continue

        print("### CREATING DIRECTORY {}".format(dir))
        new_dir = path_image_save_log + "{}".format(dir) 
        os.mkdir(new_dir) # create class dir

        for i, file in enumerate(os.listdir(path_data + dir)):

            vframes, aframes, info = read_video(path_data + "/{}/{}".format(dir,file), pts_unit='sec')
            
            for j, frame in enumerate(vframes):
                name = dir + str(i) + str(j) + ".jpg"
                path_and_name = new_dir + "/" + name
                
                print(path_and_name)

                save_frame = frame.permute(2,0,1).float()/255

                save_image(save_frame, path_and_name, padding=0)

                f.write("{}/{}/{},{}\n".format(root, dir, name, dir)) # path, label
            
    f.truncate()


print("### CONVERTED ALL FRAMES TO IMAGES SUCCESSFULLY")
print("### Saved annotations to {}.".format(path_csv_save_loc))





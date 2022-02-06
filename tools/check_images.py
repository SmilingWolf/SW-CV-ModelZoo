# Check that training images aren't truncated or damaged
# Log the ones that are to a text file

from PIL import Image
from tqdm import tqdm

images_list = open("2021_0000_0899/testlist.txt").readlines()
images_list = [r"F:\MLArchives\danbooru2021\512px\%s" % x.rstrip() for x in images_list]

for filename in tqdm(images_list):
    try:
        img = Image.open(filename)  # open the image file
        img.verify()  # verify that it is a good image, without decoding it.. quite fast
        img.close()
        img = Image.open(filename)  # open the image file
        img.transpose(
            Image.FLIP_LEFT_RIGHT
        )  # apply a simple transform to trigger all the other checks
        img.close()
    except Exception as e:
        with open("error_images.log", "a") as outfile:
            outfile.write("%s: %s\n" % (filename, str(e)))

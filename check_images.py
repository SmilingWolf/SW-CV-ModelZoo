# Check that training images aren't truncated or damaged
# Log the ones that are to a text file

from PIL import Image

images_list = open("2020_0000_0599/trainlist.txt").readlines()
images_list = [x.rstrip() for x in images_list]

for filename in images_list:
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
        with open("error_images.log", "w") as outfile:
            outfile.write("%s: %s\n" % (filename, str(e)))

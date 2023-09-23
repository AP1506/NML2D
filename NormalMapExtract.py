#Normal Map Data Extraction

from PIL import Image
import numpy as npy
import time

def image_is_rgb(map: Image):
    """
    Returns whether an image is in RGB or RGBA channel format.
    """

    channels = map.getbands()

    RGB = ('R', 'G', 'B')

    #Check that the image channels are RBG
    for x in range(min(len(channels), len(RGB))):
        if channels[x] != RGB[x]:
            return False
        
    return True

def calculate_light(normal_map: Image, diffuse_map: Image):
    """
    Return the image pixel color information as an ndarray with a shape of (pixel width, pixel height, 3)
    """

    if not image_is_rgb(normal_map):
        raise Exception("Image must have RGB channels.")
    
    if normal_map.width != diffuse_map.width or normal_map.height != diffuse_map.height:
        raise Exception("All maps must have the same dimensions.")

    lit_img_pixels = []
    normal_map_data = normal_map.getdata()
    diffuse_map_data = diffuse_map.getdata()

    r_nm = 255
    g_nm = 255
    b_nm = 255

    light_vector = (.7071, 0, .7071)
    viewer_vector = (0, 0, 1)
    reflected_vector = tuple()

    f = 1

    me_r = 0
    me_g = 0
    me_b = 0

    ga_r = .1
    ga_g = .1
    ga_b = .1

    ma_r = .1
    ma_g = .1
    ma_b = .1

    la_r = .1
    la_g = .1
    la_b = .1

    ld_r = 1
    ld_g = 1
    ld_b = 1

    ls_r = .4
    ls_g = .4
    ls_b = .4

    ms_r = 1
    ms_g = 1
    ms_b = 1

    mh = 160

    lit_row_pixels = []

    for entry in range(len(normal_map_data)):

        r_dm = diffuse_map_data[entry][0]
        g_dm = diffuse_map_data[entry][1]
        b_dm = diffuse_map_data[entry][2]

        md_r = r_dm/255
        md_g = g_dm/255
        md_b = b_dm/255

        if r_nm != normal_map_data[entry][0] or g_nm != normal_map_data[entry][1] or b_nm != normal_map_data[entry][2] or entry == 0:
            r_nm = normal_map_data[entry][0]
            g_nm = normal_map_data[entry][1]
            b_nm = normal_map_data[entry][2]
            
            #Calculate x, y, z vector components using the pixel's RGB channel values
            x = (r_nm - 128)/128
            y = (g_nm - 128)/128
            z = (b_nm - 128)/128

            normal_vector = (x, y, z)

            f = npy.dot(light_vector, normal_vector) >= 0

            #projection of light vector onto normal vector as an ndarray
            reflected_vector = npy.array(normal_vector) * ((npy.dot(light_vector, normal_vector))/npy.dot(normal_vector, normal_vector))
            #final calculation of reflected vector
            reflected_vector = tuple(npy.array(light_vector) - reflected_vector * 2)

            #calculate and add up contribution for ambient light
            r_lit = me_r + ga_r*ma_r + la_r*ma_r
            g_lit = me_g + ga_g*ma_g + la_g*ma_g
            b_lit = me_b + ga_b*ma_b + la_b*ma_b
            
            #if the surface is hit by light calculate and add its contribution
            if f:
                r_lit += (ld_r*md_r*npy.dot(light_vector, normal_vector) + ls_r*ms_r*max(0, (npy.dot(viewer_vector, reflected_vector))**mh))
                g_lit += (ld_g*md_g*npy.dot(light_vector, normal_vector) + ls_g*ms_g*max(0, (npy.dot(viewer_vector, reflected_vector))**mh))
                b_lit += (ld_b*md_b*npy.dot(light_vector, normal_vector) + ls_b*ms_b*max(0, (npy.dot(viewer_vector, reflected_vector))**mh))

            #convert values to RGB values
            r_lit = round(255*min(1, r_lit))
            g_lit = round(255*min(1, g_lit))
            b_lit = round(255*min(1, b_lit))

        lit_pixel = [r_lit, g_lit, b_lit]

        lit_row_pixels.append(lit_pixel)

        #restart the list for the next row of pixels
        if (entry + 1) % normal_map.width == 0:
            lit_img_pixels.append(lit_row_pixels.copy())

            lit_row_pixels.clear()

    return npy.array(lit_img_pixels).astype(npy.uint8)


with Image.open("Resources\Test_Normal_Map1.png") as image_info, Image.open("Resources\Test_Diffuse_Light_Map1.png") as diffuse_map_info:
    
    start = time.time()
    print("Timer started.")

    lit_image_data = calculate_light(image_info, diffuse_map_info)

    end = time.time()
    print(end - start)
    print("Timer stopped")

lit_image = Image.fromarray(lit_image_data, mode='RGB')

lit_image.save("sample.png")

a = npy.asarray(lit_image)

# print(a)

#Test if the array for the created image and the array used to create the image are the same
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         for k in range(len(a[i][j])):
#             if a[i][j][k] != lit_image_data[i][j][k]:
#                 print("The arrays are not equal")
#Normal Map Data Extraction

from PIL import Image
import numpy as npy
import time
import math

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

def vector360(numerator: int, denominator: int, z_value: float):
    """
    Return a unit vector with an angle of numerator/denominator*2*pi rads from
    positive x-axis. The z-value, or depth value, can be specified.
    """

    angle = (numerator - 1) / denominator * 2 * math.pi

    x = math.cos(angle)
    y = math.sin(angle)

    vector = (x, y, z_value)

    magnitude = sum(i**2 for i in vector)** 0.5

    #Make into unit vector
    vector = tuple((i / magnitude) for i in vector)

    return vector

def calculate_light(normal_map: Image, diffuse_map: Image, light_vector: tuple):
    """
    Return the image pixel color information as an ndarray with a shape of (pixel width, pixel height, 3)
    """

    if not image_is_rgb(normal_map):
        raise Exception("Image must have RGB channels.")
    
    if normal_map.width != diffuse_map.width or normal_map.height != diffuse_map.height:
        raise Exception("All maps must have the same dimensions.")

    lit_img_pixels = []
    normal_map_data = tuple(normal_map.getdata())
    diffuse_map_data = tuple(diffuse_map.getdata())

    calculated_values = {}

    r_nm = g_nm = b_nm = 255

    #light_vector = (.7071, 0, .7071)
    viewer_vector = (0, 0, 1)
    reflected_vector = tuple()

    #material light emission values
    me_r = me_g = me_b = 0

    #global ambient light values
    ga_r = ga_g = ga_b = .1

    #material ambient light values
    ma_r = ma_g = ma_b = .1

    #light's ambient intensity values
    la_r = la_g = la_b = .1

    #light's diffuse intensity values
    ld_r = ld_g = ld_b = 1

    #light's specular intensity values
    ls_r = ls_g = ls_b = .4

    #material specular light values
    ms_r = ms_g = ms_b = 1

    mh = 160

    lit_row_pixels = []

    for entry in range(len(normal_map_data)):

        r_dm = diffuse_map_data[entry][0]
        g_dm = diffuse_map_data[entry][1]
        b_dm = diffuse_map_data[entry][2]

        md_r = r_dm/255
        md_g = g_dm/255
        md_b = b_dm/255

        r_nm = normal_map_data[entry][0]
        g_nm = normal_map_data[entry][1]
        b_nm = normal_map_data[entry][2]

        dict_key = ((r_dm, g_dm, b_dm), (r_nm, g_nm, b_nm))

        #if the color has not been used for calculations yet, calculate and record values
        if dict_key not in calculated_values:
            
            #Calculate x, y, z vector components using the pixel's RGB channel values
            x = (r_nm - 128)/128
            y = (g_nm - 128)/128
            z = (b_nm - 128)/128

            normal_vector = (x, y, z)

            #Projection of light vector onto normal vector as an ndarray.
            reflected_vector = tuple(vector * ((npy.dot(light_vector, normal_vector))/sum(i*i for i in normal_vector)) for vector in normal_vector)
            #Final calculation of reflected vector.
            #Note that the reflected vector will be a unit vector if the light vector is one.
            reflected_vector = tuple(x1 - x2 for (x1, x2) in zip(light_vector,tuple(x * 2 for x in reflected_vector)))

            #calculate and add up contribution for ambient light
            r_lit = me_r + ga_r*ma_r + la_r*ma_r
            g_lit = me_g + ga_g*ma_g + la_g*ma_g
            b_lit = me_b + ga_b*ma_b + la_b*ma_b
            
            #if the surface is hit by light calculate and add its contribution
            if npy.dot(light_vector, normal_vector) >= 0:
                r_lit += (ld_r*md_r*npy.dot(light_vector, normal_vector) + ls_r*ms_r*max(0, (npy.dot(viewer_vector, reflected_vector))**mh))
                g_lit += (ld_g*md_g*npy.dot(light_vector, normal_vector) + ls_g*ms_g*max(0, (npy.dot(viewer_vector, reflected_vector))**mh))
                b_lit += (ld_b*md_b*npy.dot(light_vector, normal_vector) + ls_b*ms_b*max(0, (npy.dot(viewer_vector, reflected_vector))**mh))

            #convert values to RGB values
            r_lit = round(255*min(1, r_lit))
            g_lit = round(255*min(1, g_lit))
            b_lit = round(255*min(1, b_lit))

            calculated_values[dict_key] = (r_lit, g_lit, b_lit)

        else:
            r_lit = calculated_values.get(dict_key)[0]
            g_lit = calculated_values.get(dict_key)[1]
            b_lit = calculated_values.get(dict_key)[2]
        
        lit_pixel = [r_lit, g_lit, b_lit]

        lit_row_pixels.append(lit_pixel)

        #restart the list for the next row of pixels
        if (entry + 1) % normal_map.width == 0:
            lit_img_pixels.append(lit_row_pixels.copy())

            lit_row_pixels.clear()

    return npy.array(lit_img_pixels).astype(npy.uint8)


with Image.open("./Resources/Test_Normal_Map1.png") as image_info, Image.open("./Resources/Test_Diffuse_Light_Map1.png") as diffuse_map_info:
    
    num_frames = 5

    for x in range(1, num_frames + 1):
        start = time.time()
        print("Timer started.")

        light_vector = vector360(x, num_frames, 1)

        lit_image_data = calculate_light(image_info, diffuse_map_info, light_vector)

        end = time.time()
        print(end - start)
        print("Timer stopped")

        lit_image = Image.fromarray(lit_image_data, mode='RGB')

        lit_image.save("Output/sample" + str(x) + ".png")

        print("Finished rendering frame " + str(x) + " out of " + str(num_frames))

#a = npy.asarray(lit_image)

# print(a)

#Test if the array for the created image and the array used to create the image are the same
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         for k in range(len(a[i][j])):
#             if a[i][j][k] != lit_image_data[i][j][k]:
#                 print("The arrays are not equal")
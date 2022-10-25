from PIL import Image
basewidth = 300
file_path = 'private_materials/data/' + merged_df[merged_df.height == 542.0].file_id.values[0] + '.jpg'
img = Image.open(file_path)
# wpercent = (basewidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('original.jpg')

new_width = 640
new_height = 320
img = img.resize((new_width, new_height), Image.ANTIALIAS)

img.save('test.jpg')
[[0,0][640, 0][640, 542][0, 542]]
width, height = img.size
new_height = 320
new_width = 640
aspect_ratio_height = height / new_height
aspect_ratio_width = width / new_width
initial_corners = np.array(points, dtype=np.int32)
new_array = []
for array in initial_corners:
    array_length = len(array)
    for item in array:
        test_array = [item[0] * aspect_ratio_width, item[1]* aspect_ratio_height]
        new_array.append(test_array)


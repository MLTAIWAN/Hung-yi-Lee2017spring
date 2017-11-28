from PIL import Image

lena = Image.open('lena.png')
# lena.show()

lena_modified = Image.open('lena_modified.png')
# lena_modified.show()

width,heigth = lena.size
ans = Image.new(mode='RGBA',size=lena.size)

lena_color = lena.load()
lena_modified_color = lena_modified.load()
ans_color = ans.load()

for i in range(0,width):
    for j in range(0,heigth):
        lena_data = lena_color[i,j]
        lena_modified_data = lena_modified_color[i,j]
        if lena_data != lena_modified_data:
            ans_color[i,j] = lena_modified_data

ans.show()

ans.save('ans.png')




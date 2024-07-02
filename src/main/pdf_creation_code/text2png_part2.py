from PIL import Image, ImageFont, ImageDraw
import glob
img = Image.new('RGB', (1080, 1980), color = (255,255,255))
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)
LC_path = "/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/45_lc_cases/tolerance-level-cases/only-39_multi_goods"

print("lc path:")
print(LC_path)

def replace_tabs(text, tab_size=4):
    return text.replace('\t', ' ' * tab_size)


for lc in glob.glob(f"{LC_path}/*.txt"):
	lc_prefix = lc.split("/")[-1].split(".txt")[0]
	with open(lc, "r") as file:
		data = file.readlines()
		print(len(data))
		pre_index = 0
		remaining_data = ""
		if len(data) > 1:
			for index in range(0,len(data)+1,min([40, len(data)])):
				print(f"index: {index}")
				img = Image.new('RGB', (1300, 2180), color = (255,255,255))
				new_data = "".join(data[pre_index:pre_index+index])
				new_data = replace_tabs(new_data)
				pre_index = index
				if index == 0:
					continue
				ImageDraw.Draw(img).text((0,0), new_data, font=fnt, fill=(0,0,0))
				img.save(f"lc_{lc_prefix}_{index}.png")
				remaining_data = data[index:]

			
			img = Image.new('RGB', (1300, 2180), color = (255,255,255))
			new_data = "".join(remaining_data)
			new_data = replace_tabs(new_data)
			ImageDraw.Draw(img).text((0,0), new_data, font=fnt, fill=(0,0,0))
			img.save(f"lc_remaining.png")


		else:
			img = Image.new('RGB', (1300, 2180), color = (255,255,255))
			new_data = "".join(data[pre_index:])
			ImageDraw.Draw(img).text((0,0), new_data, font=fnt, fill=(0,0,0))
			img.save(f"lc_{lc_prefix}.png")

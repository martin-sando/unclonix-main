import os
import PIL.Image as Image

folder_in = "in2"
folder_out = "in11"
quality_val = 95
compression_val = 0.25

input_files = sorted(os.listdir(folder_in))

for i, file in enumerate(input_files):
    print(i, file)
    image = Image.open(f"{folder_in}/{file}")
    image = image.resize((int(image.width * compression_val), int(image.height * compression_val)), Image.LANCZOS)  # LANCZOS as of Pillow 2.7
    # Followed by the save method
    image.save(f"{folder_out}/{file}", 'JPEG', quality=quality_val)
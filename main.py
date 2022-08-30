import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import os

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

while True:
    try:
        prompt = input("Enter a prompt: ")
        d = prompt[:80]
        while True:
            try:
                count = input("Enter a number of images to produce (default 10): ")
                if count == "" or count.isspace():
                    count = 10
                else:
                    count = int(count)
                break
            except ValueError:
                continue
            except KeyboardInterrupt:
                print("Thanks!")
                exit()

        os.makedirs(f"results/{d}", exist_ok=True)

        files = os.listdir(f"results/{d}")
        if len(files) > 0:
            image_id = max([int(f.split(".")[0]) for f in files]) + 1
        else:
            image_id = 0
        for i in range(image_id, image_id + count):
            image = pipe(prompt)["sample"][0]
            image.save(f"results/{d}/{i}.png")
        print(f"Done! Generated {count} images to \"results/{d}\"")
    except KeyboardInterrupt:
        continue
# Opening a can of worms

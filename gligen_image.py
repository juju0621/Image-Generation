import os
import json
import torch
from tqdm import tqdm
from diffusers.utils import load_image
from argparse import Namespace, ArgumentParser
from diffusers import StableDiffusionGLIGENPipeline


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--label_path", type=str,
                        default="/content/drive/MyDrive/cvpdl_hw3/label.json")
    parser.add_argument("--images_dir", type=str,
                        default="./images_512")
    parser.add_argument("--model_name_or_path", type=str,
                        default="anhnct/Gligen_Text_Image",
                        help="model name or path")
    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help="output dir")
    return parser.parse_args()

def normalize_bounding_boxes(absolute_box, image_width, image_height):
    normalized_box = [
            absolute_box[0] / image_width,
            absolute_box[1] / image_height,
            absolute_box[2] / image_width,  
            absolute_box[3] / image_height 
        ]
    return normalized_box


def main():
    args = parse_arguments()
    
    label_dir = args.label_path
    os.makedirs(args.output_dir, exist_ok=True)
    with open(label_dir, "r") as f:
        labels = json.load(f)
    
    
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        args.model_name_or_path, variant="fp16", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    for label in tqdm(labels):
        image_name = label["image"]
        height = label["height"]
        width = label["width"]
        prompt = label["prompt_w_suffix"]
        boxes = [normalize_bounding_boxes(box, width, height) for box in label["bboxes"]]
        phrases = label["labels"]
        
        image_path = os.path.join(args.images_dir, image_name)
        image = load_image(image_path)
        
        images = pipe(
            prompt=prompt,
            gligen_phrases=phrases,
            gligen_images=image,
            gligen_boxes=boxes,
            gligen_scheduled_sampling_beta=1,
            height=height,
            width=width,
            output_type="pil",
            num_inference_steps=100,
        ).images
        
        output_path = os.path.join(args.output_dir, image_name)
        images[0].save(output_path)

if __name__ == "__main__":
    main()  
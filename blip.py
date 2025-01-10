import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from argparse import Namespace, ArgumentParser
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str,
                        default="/content/drive/MyDrive/cvpdl_hw3/images")
    parser.add_argument("--label_path", type=str,
                        default="/content/drive/MyDrive/cvpdl_hw3/label.json")
    parser.add_argument("--model_name_or_path", type=str,
                        default="Salesforce/blip2-opt-2.7b",
                        help="model name or path")
    parser.add_argument("--output_path", type=str,
                        default=None,
                        help="output path")
    parser.add_argument("--use_8bit", type=bool,
                        default=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    image_dir = args.image_dir
    label_dir = args.label_path

    with open(label_dir, "r") as f:
        labels = json.load(f)

    bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit)

    processor = Blip2Processor.from_pretrained(args.model_name_or_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_caption = {}

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        prompt = "Question: please describe the photo in detail Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
        generated_ids = model.generate(**inputs, max_length=64)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        image_caption[image_file] = generated_text

    result = []
    for pic in tqdm(labels, desc="Creatin Captions"):
        image_file = pic["image"]
        labels = pic["labels"]
        height = pic["height"]
        width = pic["width"]
        bboxes = pic["bboxes"]
        generated_text = image_caption[image_file].capitalize()
        prompt_w_label = generated_text + ". " + ", ".join(list(set(labels))) + f", height: {height}, width:{width}"
        prompt_w_suffix = prompt_w_label + ", HD quality, highly detailed"
        d = {
            "image": image_file,
            "labels": labels,
            "height": height,
            "width": width,
            "bboxes": bboxes,
            "generated_text": generated_text,
            "prompt_w_label": prompt_w_label,
            "prompt_w_suffix": prompt_w_suffix
        }
        result.append(d)

    with open(args.output_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
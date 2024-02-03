import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/MoE-LLaVA-hf')

import torch
from PIL import Image
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def inference(image, inp, tokenizer=None, model=None, processor=None):
    image_processor = processor['image']
    conv_mode = "phi"  # phi or qwen or stablelm
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(Image.open(image).convert('RGB'), return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)
    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return outputs

class Predictor(BasePredictor):
    def setup(self) -> None:
        disable_torch_init()
        model_path = 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384'
        device = 'cuda'
        load_4bit, load_8bit = False, False
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        input_text: str = Input(default="What is unusual about this image?"),
    ) -> str:
        out_string = inference(input_image, input_text, self.tokenizer, self.model, self.processor)
        return out_string
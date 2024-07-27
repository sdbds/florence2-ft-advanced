import os
from torch.utils.data import Dataset, random_split
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, split, task_prompt):
        self._split = split
        self.data = []
        self.task_prompt = task_prompt

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text

class CustomDataset(BaseDataset):
    def __init__(self, images_dir, texts_dir, task_prompt, split=None):
        super().__init__(split, task_prompt)
        self.data = self.load_data(images_dir, texts_dir)
        if split:
            train_size = int(0.8 * len(self.data))
            val_size = len(self.data) - train_size
            self.train_data, self.val_data = random_split(self.data, [train_size, val_size])
            self.data = self.train_data if split == 'train' else self.val_data

    def load_data(self, images_dir, texts_dir):
        data = []
        if texts_dir:
            images = sorted(os.listdir(images_dir))
            texts = sorted(os.listdir(texts_dir))

            for image_file, text_file in zip(images, texts):
                if image_file.endswith(('.png', '.jpg', '.jpeg', '.webp')) and text_file.endswith('.txt'):
                    with open(os.path.join(texts_dir, text_file), 'r', encoding='utf-8') as file:
                        response = file.read().strip()
                    try:
                        with Image.open(os.path.join(root, image_file)) as img:
                            # 尝试加载整个图片
                            img.load()
                        del img

                        data.append({
                            "image_path": os.path.join(root, image_file),
                            "response": response
                        })
                    except Exception as e:
                        print(f"Error loading image {image_file}: {e}")
        else:
            for root, dirs, files in os.walk(images_dir):
                for image_file in files:
                    if image_file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        text_file = os.path.splitext(image_file)[0] + '.txt'
                        text_path = os.path.join(root, text_file)
                        if os.path.exists(text_path):
                            with open(text_path, 'r', encoding='utf-8') as file:
                                response = file.read().strip()
                            try:
                                with Image.open(os.path.join(root, image_file)) as img:
                                    # 尝试加载整个图片
                                    img.load()
                                del img

                                data.append({
                                    "image_path": os.path.join(root, image_file),
                                    "response": response
                                })
                            except Exception as e:
                                print(f"Error loading image {image_file}: {e}")
        return data

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.task_prompt
        answer = self.correct_casing_finqa(example["response"])
        image = Image.open(example["image_path"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image

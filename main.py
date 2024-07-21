import os
import torch
import logging
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from dataset import CustomDataset
from train import train_model
from config import parse_args

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args):
    task_prompts = {
        "<CAPTION>": "What does the image describe?",
        "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
        "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
    }
    task_prompt = task_prompts[args.task_type]

    logger.info("==========Starting==========")
    logger.info("Settings:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    logger.info("=======Loading Dataset=======")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    dataset = CustomDataset(
        images_dir=args.images_dir,
        texts_dir=args.texts_dir,
        task_prompt=task_prompt,
        split=None,
    )
    total_size = len(dataset.data)
    train_size = int(args.train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.split = "train"
    val_dataset.dataset.split = "validation"

    logger.info(f"Total dataset: {total_size}")
    logger.info(f"Training dataset: {train_size}")
    logger.info(f"Validation dataset: {val_size}")

    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(
            text=list(questions), images=list(images), return_tensors="pt", padding=True
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        return inputs, answers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=args.persistent_data_loader_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * len(train_loader)

    train_model(train_loader, val_loader, model, processor, args, device, logger)


if __name__ == "__main__":
    args = parse_args()
    main(args)

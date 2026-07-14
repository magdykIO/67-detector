from fastai.vision.all import *

def train():
    path = Path("fastai")

    meme_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock), # Images will turn into categories
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        
        item_tfms=Resize(224),
        batch_tfms=aug_transforms()
    )

    dataloaders = meme_block.dataloaders(path, bs=16)
    print("categories:", dataloaders.vocab)

    learn = vision_learner(dataloaders, resnet18, metrics=error_rate) # Using resnet18 model, looking at error_rate
    learn.fine_tune(3) # Look through the data 3 times
    learn.export('67.pkl')

if __name__ == '__main__':
    train()
import time
from ddgs import DDGS
from fastai.vision.all import *


def parser():
    meme = {
        "sixseven": "67 meme NO FACES only numbers itself",
        "not_sixseven": "real world photos OR telegram chat screen OR python code editor screenshot"
    }

    for request, search_query in meme.items():
        path = Path(f"meme/{request}")
        path.mkdir(exist_ok=True, parents=True)

        results = DDGS().images(
            query=search_query,
            size="Small",
            max_results=50 
            # Add size filter
        )
        time.sleep(3)

        only_urls = [res.get('image') for res in results]
        download_images(path, urls=only_urls)

        # Check downloaded images
        failed = verify_images(get_image_files(path))
        failed.map(Path.unlink)

if __name__ == '__main__':
    parser()
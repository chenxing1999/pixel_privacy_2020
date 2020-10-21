import torch
from torch import optim
from torchvision import transforms
from PIL import Image


import logging
import tqdm


class SimpleWhiteBox(object):
    def __init__(
        self,
    ):
        self.default_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.default_reverse = transforms.Compose(
            [
                transforms.Normalize([0,0,0], [2, 2, 2]),
                transforms.Normalize([-0.5, -0.5, -0.5], [1, 1, 1]),
                self.make_image,
                transforms.ToPILImage(),
            ]
        )
        self.logger = logging.getLogger(__name__)

    def make_image(self, tensor):
        tensor = tensor.clamp(0, 1)
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        return tensor


    def attack_binary(
        self, 
        model, 
        image_path,
        transform=None,
        reverse_transform=None,
        lr=1e-7,
        device=None,
        ilen=5,
        jlen=20,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tensor = self.load_attack_image(image_path, transform)
        
        # Calculate original distribution of data
        no_color_channel = [0, 2, 3]
        original_std = tensor.std(no_color_channel).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        original_mean = tensor.mean(no_color_channel).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)

        optimizer = None
        
        model.to(device)
        model.eval()
        tensor = tensor.to(device)

        self.logger.debug("Finish prepare data")

        for i in range(ilen):
            for j in tqdm.tqdm(range(jlen), ascii=True):
                tensor.requires_grad_(True)
                optimizer = torch.optim.Adam([tensor], lr)

                model_output = model(tensor)

                target = torch.zeros_like(model_output)
                target[0, 0] = 1
                optimizer.zero_grad()
                model_output.backward(target)
                optimizer.step()

            # Back to i loop. Convert tensor to original distribution
            with torch.no_grad():
                if not optimizer is None:
                    tensor = optimizer.param_groups[0]["params"][0]

                # Convert current tensor back to original dist
                tensor = tensor.sub(
                        tensor.mean(no_color_channel).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    )
                tensor = (
                    tensor.div(
                        tensor.std(no_color_channel).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    )
                    .mul(original_std)
                    .add(original_mean)
                )
                tensor = tensor.cpu()
                image = self.reverse(tensor[0], reverse_transform)
                tensor = self.transform(image).unsqueeze(0)
                tensor = tensor.to(device)

                score = model(tensor)[0].item()
                self.logger.info(f"Score: {score}")
                if score < 50:
                    return image
        return image


    def load_attack_image(self, image_path, transform=None):
        image = Image.open(image_path)
        tensor = self.transform(image, transform)
        # For inference
        tensor = tensor.unsqueeze(0)
        return tensor

    def transform(self, image, transform=None):
        if transform:
            tensor = transform(image)
        else:
            tensor = self.default_transform(image)
        return tensor
        

    def reverse(self, tensor, reverse_transform=None):
        if reverse_transform:
            return reverse_transform(tensor)
        return self.default_reverse(tensor)

if __name__ == "__main__":
    image_path = "../../../2020-Pixel-Privacy-Task-master/pp2020_test/Places365_val_00015483.png"
    image = Image.open(image_path)
    original_tensor = transforms.ToTensor()(image)
    
    attacker = SimpleWhiteBox()

    tensor = attacker.default_transform(image)
    rev_tensor = attacker.default_reverse(tensor)

    import pdb; pdb.set_trace()



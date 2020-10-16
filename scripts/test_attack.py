# LAZY IMPORT
import os
import sys
cur_dir = os.path.dirname(__file__)
master_dir = os.path.join("..", cur_dir)
sys.path.append(master_dir)
##############################


from pixel_privacy.models.biqa_model import BIQAModel
from pixel_privacy.attacks.simple_white_box import SimpleWhiteBox



if __name__ == "__main__":
   weight_path = "../../2020-Pixel-Privacy-Task-master/BIQA_model/KonCept512.pth"
   image_path = "../../2020-Pixel-Privacy-Task-master/pp2020_test/Places365_val_00015483.png"

   model = BIQAModel(weight_path, pretrained=None)
   print("Inited model")
   attack = SimpleWhiteBox()

   attack.attack_binary(
        model,
        image_path,
        device="cuda",
        lr=1e-2,
    )

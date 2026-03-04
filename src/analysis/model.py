"""
TorchVision model loading and inference for animal classification.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet class indices that correspond to animals we want to detect
# See: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
IMAGENET_TO_ANIMAL = {
    # Birds
    7: "bird",      # cock
    8: "bird",      # hen
    9: "bird",      # ostrich
    10: "bird",     # brambling
    11: "bird",     # goldfinch
    12: "bird",     # house finch
    13: "bird",     # junco
    14: "bird",     # indigo bunting
    15: "bird",     # robin
    16: "bird",     # bulbul
    17: "bird",     # jay
    18: "bird",     # magpie
    19: "bird",     # chickadee
    20: "bird",     # water ouzel
    21: "bird",     # kite
    22: "bird",     # bald eagle
    23: "bird",     # vulture
    24: "bird",     # great grey owl
    80: "bird",     # black grouse
    81: "bird",     # ptarmigan
    82: "bird",     # ruffed grouse
    83: "bird",     # prairie chicken
    84: "bird",     # peacock
    85: "bird",     # quail
    86: "bird",     # partridge
    87: "bird",     # african grey
    88: "bird",     # macaw
    89: "bird",     # sulphur-crested cockatoo
    90: "bird",     # lorikeet
    91: "bird",     # coucal
    92: "bird",     # bee eater
    93: "bird",     # hornbill
    94: "bird",     # hummingbird
    95: "bird",     # jacamar
    96: "bird",     # toucan
    97: "bird",     # drake
    98: "bird",     # red-breasted merganser
    99: "bird",     # goose
    100: "bird",    # black swan
    127: "bird",    # white stork
    128: "bird",    # black stork
    129: "bird",    # spoonbill
    130: "bird",    # flamingo
    131: "bird",    # little blue heron
    132: "bird",    # american egret
    133: "bird",    # bittern
    134: "bird",    # crane
    135: "bird",    # limpkin
    136: "bird",    # european gallinule
    137: "bird",    # american coot
    138: "bird",    # bustard
    139: "bird",    # ruddy turnstone
    140: "bird",    # red-backed sandpiper
    141: "bird",    # redshank
    142: "bird",    # dowitcher
    143: "bird",    # oystercatcher
    144: "bird",    # pelican
    145: "bird",    # king penguin
    146: "bird",    # albatross
    
    # Cats
    281: "cat",     # tabby
    282: "cat",     # tiger cat
    283: "cat",     # persian cat
    284: "cat",     # siamese cat
    285: "cat",     # egyptian cat
    286: "cat",     # cougar (mountain lion)
    287: "cat",     # lynx
    288: "cat",     # leopard
    289: "cat",     # snow leopard
    290: "cat",     # jaguar
    291: "cat",     # lion
    292: "cat",     # tiger
    293: "cat",     # cheetah
    
    # Dogs
    151: "dog",     # chihuahua
    152: "dog",     # japanese spaniel
    153: "dog",     # maltese dog
    154: "dog",     # pekinese
    155: "dog",     # shih-tzu
    156: "dog",     # blenheim spaniel
    157: "dog",     # papillon
    158: "dog",     # toy terrier
    159: "dog",     # rhodesian ridgeback
    160: "dog",     # afghan hound
    161: "dog",     # basset
    162: "dog",     # beagle
    163: "dog",     # bloodhound
    164: "dog",     # bluetick
    165: "dog",     # black-and-tan coonhound
    166: "dog",     # walker hound
    167: "dog",     # english foxhound
    168: "dog",     # redbone
    169: "dog",     # borzoi
    170: "dog",     # irish wolfhound
    171: "dog",     # italian greyhound
    172: "dog",     # whippet
    173: "dog",     # ibizan hound
    174: "dog",     # norwegian elkhound
    175: "dog",     # otterhound
    176: "dog",     # saluki
    177: "dog",     # scottish deerhound
    178: "dog",     # weimaraner
    179: "dog",     # staffordshire terrier
    180: "dog",     # american staffordshire terrier
    181: "dog",     # bedlington terrier
    182: "dog",     # border terrier
    183: "dog",     # kerry blue terrier
    184: "dog",     # irish terrier
    185: "dog",     # norfolk terrier
    186: "dog",     # norwich terrier
    187: "dog",     # yorkshire terrier
    188: "dog",     # wire-haired fox terrier
    189: "dog",     # lakeland terrier
    190: "dog",     # sealyham terrier
    191: "dog",     # airedale
    192: "dog",     # cairn
    193: "dog",     # australian terrier
    194: "dog",     # dandie dinmont
    195: "dog",     # boston bull
    196: "dog",     # miniature schnauzer
    197: "dog",     # giant schnauzer
    198: "dog",     # standard schnauzer
    199: "dog",     # scotch terrier
    200: "dog",     # tibetan terrier
    201: "dog",     # silky terrier
    202: "dog",     # soft-coated wheaten terrier
    203: "dog",     # west highland white terrier
    204: "dog",     # lhasa
    205: "dog",     # flat-coated retriever
    206: "dog",     # curly-coated retriever
    207: "dog",     # golden retriever
    208: "dog",     # labrador retriever
    209: "dog",     # chesapeake bay retriever
    210: "dog",     # german short-haired pointer
    211: "dog",     # vizsla
    212: "dog",     # english setter
    213: "dog",     # irish setter
    214: "dog",     # gordon setter
    215: "dog",     # brittany spaniel
    216: "dog",     # clumber
    217: "dog",     # english springer
    218: "dog",     # welsh springer spaniel
    219: "dog",     # cocker spaniel
    220: "dog",     # sussex spaniel
    221: "dog",     # irish water spaniel
    222: "dog",     # kuvasz
    223: "dog",     # schipperke
    224: "dog",     # groenendael
    225: "dog",     # malinois
    226: "dog",     # briard
    227: "dog",     # kelpie
    228: "dog",     # komondor
    229: "dog",     # old english sheepdog
    230: "dog",     # shetland sheepdog
    231: "dog",     # collie
    232: "dog",     # border collie
    233: "dog",     # bouvier des flandres
    234: "dog",     # rottweiler
    235: "dog",     # german shepherd
    236: "dog",     # doberman
    237: "dog",     # miniature pinscher
    238: "dog",     # greater swiss mountain dog
    239: "dog",     # bernese mountain dog
    240: "dog",     # appenzeller
    241: "dog",     # entlebucher
    242: "dog",     # boxer
    243: "dog",     # bull mastiff
    244: "dog",     # tibetan mastiff
    245: "dog",     # french bulldog
    246: "dog",     # great dane
    247: "dog",     # saint bernard
    248: "dog",     # eskimo dog
    249: "dog",     # malamute
    250: "dog",     # siberian husky
    251: "dog",     # dalmatian
    252: "dog",     # affenpinscher
    253: "dog",     # basenji
    254: "dog",     # pug
    255: "dog",     # leonberg
    256: "dog",     # newfoundland
    257: "dog",     # great pyrenees
    258: "dog",     # samoyed
    259: "dog",     # pomeranian
    260: "dog",     # chow
    261: "dog",     # keeshond
    262: "dog",     # brabancon griffon
    263: "dog",     # pembroke
    264: "dog",     # cardigan
    265: "dog",     # toy poodle
    266: "dog",     # miniature poodle
    267: "dog",     # standard poodle
    268: "dog",     # mexican hairless
    269: "dog",     # timber wolf (also wildlife)
    270: "dog",     # white wolf
    271: "dog",     # red wolf
    273: "dog",     # dingo
    274: "dog",     # dhole
    275: "dog",     # african hunting dog
    
    # Squirrels and small rodents
    335: "squirrel",  # fox squirrel
    336: "squirrel",  # marmot
    
    # Foxes
    277: "fox",     # red fox
    278: "fox",     # kit fox
    279: "fox",     # arctic fox
    280: "fox",     # grey fox
    
    # Rabbits and hares
    330: "rabbit",  # wood rabbit
    331: "rabbit",  # hare
    332: "rabbit",  # angora
    
    # Other mammals
    334: "hedgehog",  # hedgehog
    337: "beaver",    # beaver
    338: "hamster",   # hamster
    
    # Deer
    351: "deer",    # hartebeest
    352: "deer",    # impala
    353: "deer",    # gazelle
    
    # Rodents/pests
    333: "mouse",   # hamster predecessor
}

# Bird species commonly found in UK gardens and woodlands.
# ImageNet classes outside this set are still detected as "bird"
# but won't receive a species label.
IMAGENET_TO_BIRD_SPECIES: dict[int, str] = {
    8: "hen",
    10: "brambling",
    11: "goldfinch",
    12: "house finch",
    15: "robin",
    17: "jay",
    18: "magpie",
    19: "chickadee",
    20: "water ouzel",
    21: "kite",
    24: "great grey owl",
    80: "black grouse",
    81: "ptarmigan",
    84: "peacock",
    85: "quail",
    86: "partridge",
    97: "drake",
    99: "goose",
}

# Simplified animal classes for output
ANIMAL_CLASSES = [
    "bird",
    "cat", 
    "dog",
    "squirrel",
    "fox",
    "rabbit",
    "deer",
    "hedgehog",
    "mouse",
    "hamster",
    "beaver",
    "unknown",
]


class ModelLoader:
    """
    Loads and manages TorchVision models for inference.
    
    Uses MobileNetV3 by default for efficiency on Raspberry Pi.
    """
    
    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        device: Optional[str] = None,
        weights_path: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        
        self._model: Optional[nn.Module] = None
        self._transform: Optional[transforms.Compose] = None
        
        logger.info(f"ModelLoader initialized: {model_name} on {self.device}")
    
    def load(self) -> nn.Module:
        """Load the model with pretrained weights."""
        if self._model is not None:
            return self._model
        
        logger.info(f"Loading model: {self.model_name}")
        
        if self.model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self._model = models.mobilenet_v3_small(weights=weights)
            self._transform = weights.transforms()
        elif self.model_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            self._model = models.mobilenet_v3_large(weights=weights)
            self._transform = weights.transforms()
        elif self.model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self._model = models.resnet18(weights=weights)
            self._transform = weights.transforms()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Load custom weights if provided
        if self.weights_path and self.weights_path.exists():
            logger.info(f"Loading custom weights from: {self.weights_path}")
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
        
        self._model = self._model.to(self.device)
        self._model.eval()
        
        logger.info(f"Model loaded successfully")
        return self._model
    
    def get_transform(self) -> transforms.Compose:
        """Get the preprocessing transform for the model."""
        if self._transform is None:
            self.load()
        return self._transform
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess an image for model input."""
        transform = self.get_transform()
        tensor = transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image: Image.Image) -> tuple[int, float]:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Tuple of (class_index, confidence)
        """
        model = self.load()
        input_tensor = self.preprocess(image)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        class_idx = probabilities.argmax().item()
        confidence = probabilities[class_idx].item()
        
        return class_idx, confidence
    
    def predict_top_k(self, image: Image.Image, k: int = 5) -> list[tuple[int, float]]:
        """
        Get top-k predictions for an image.
        
        Args:
            image: PIL Image to classify
            k: Number of top predictions to return
            
        Returns:
            List of (class_index, confidence) tuples
        """
        model = self.load()
        input_tensor = self.preprocess(image)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        top_probs, top_indices = probabilities.topk(k)
        
        return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model loading
    loader = ModelLoader()
    model = loader.load()
    print(f"Model loaded: {loader.model_name}")
    print(f"Device: {loader.device}")
    
    # Test with a dummy image
    dummy_image = Image.new("RGB", (224, 224), color="green")
    class_idx, confidence = loader.predict(dummy_image)
    print(f"Prediction: class {class_idx}, confidence {confidence:.4f}")
    
    # Check if it's an animal
    if class_idx in IMAGENET_TO_ANIMAL:
        animal = IMAGENET_TO_ANIMAL[class_idx]
        print(f"Detected animal: {animal}")
    else:
        print("No animal detected")
    
    loader.unload()

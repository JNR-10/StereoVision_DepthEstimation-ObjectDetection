import argparse
import torch
import torch.nn as nn
import cv2

import torchvision.transforms as transforms
from PIL import Image

from flopth import flopth
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchstat import stat
import numpy as np

import config
from networks.HitNet.models import HITNet


class HitNetEstimator:
    def get_config(self):
        parser = argparse.ArgumentParser(description='HITNet')
        parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
        parser.add_argument('--fea_c', type=list, default=[32, 24, 24, 16, 16], help='feature extraction channels')
        # parse arguments, set seeds
        args = parser.parse_args()
        return args
    def __init__(self):
        self.model = HITNet(self.get_config())
        print(self.model)
        self.model = nn.DataParallel(self.model)
        # Load checkpoint with compatibility for CPU-only machines.
        try:
            if torch.cuda.is_available():
                print("Loading HITNet checkpoint on CUDA device...")
                state_dict = torch.load(config.HITNET_MODEL_PATH)
            else:
                print("CUDA not available — loading HITNet checkpoint to CPU (map_location='cpu')...")
                state_dict = torch.load(config.HITNET_MODEL_PATH, map_location=torch.device('cpu'))
        except Exception as e:
            # Fallback: try forcing CPU map_location in case the first attempt failed
            print(f"Warning: failed to load checkpoint normally ({e}). Retrying with map_location='cpu'.")
            state_dict = torch.load(config.HITNET_MODEL_PATH, map_location=torch.device('cpu'))

        # Some checkpoints store the model dict under 'model' key; others may be the state_dict itself.
        model_state = state_dict.get('model', state_dict) if isinstance(state_dict, dict) else state_dict

        try:
            self.model.load_state_dict(model_state)
        except RuntimeError:
            # Try to be resilient to 'module.' prefix mismatches (DataParallel vs single-GPU saves)
            new_state = {}
            for k, v in model_state.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else 'module.' + k
                new_state[new_key] = v
            try:
                self.model.load_state_dict(new_state)
            except Exception as e:
                print('Failed to load HITNet state_dict into model:', e)
                raise

        # Determine runtime device and move model there.
        # If config requests CUDA but CUDA isn't available or PyTorch wasn't built with CUDA,
        # fall back to CPU to avoid AssertionError later when tensors are moved.
        try:
            desired = config.DEVICE if isinstance(config.DEVICE, str) else str(config.DEVICE)
        except Exception:
            desired = str(getattr(config, 'DEVICE', 'cpu'))

        if desired.startswith('cuda') and not torch.cuda.is_available():
            print(f"Warning: config.DEVICE='{desired}' requests CUDA but CUDA is not available; using CPU instead.")
            self.device = torch.device('cpu')
        else:
            try:
                self.device = torch.device(desired)
            except Exception:
                print(f"Warning: invalid config.DEVICE='{desired}'; falling back to CPU.")
                self.device = torch.device('cpu')

        try:
            self.model.to(self.device)
        except Exception:
            print(f"Warning: could not move model to {self.device}, falling back to 'cpu'.")
            self.device = torch.device('cpu')
            self.model.to(self.device)

    def profile(self):
        print("Profiling Architecture : {}".format(config.ARCHITECTURE))
        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))
        model = HITNet(self.get_config())
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))

        dummy_inputs = torch.rand(1, 3, width, height)
        print("=====START Profile With FLOPTH========")
        flops, params = flopth(model, inputs=(dummy_inputs,dummy_inputs))
        print("With flopth -> FLOPS: {}, params: {}".format(flops, params))
        print("=====END Profile With FLOPTH========")
    def get_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')
    def preprocess(self, left_image, right_image):
        left_img = self.load_image(left_image)
        right_img = self.load_image(right_image)

        w, h = left_img.size

        # normalize
        processed = self.get_transform()
        left_img = processed(left_img).numpy()
        right_img = processed(right_img).numpy()

        # pad to size 1280x384
        top_pad = 384 - h
        right_pad = 1280 - w
        assert top_pad > 0 and right_pad > 0
        # pad images
        left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                               constant_values=0)
        return torch.from_numpy(left_img).unsqueeze(0), torch.from_numpy(right_img).unsqueeze(0)
        """
        imgL = cv2.imread(left_image, cv2.IMREAD_COLOR)
        imgR = cv2.imread(right_image, cv2.IMREAD_COLOR)

        input_height, input_width = imgL.shape[:2]

        imgL = cv2.resize(imgL, (input_width, input_height))
        imgR = cv2.resize(imgR, (input_width, input_height))

        # Shape (1, 6, None, None)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

        imgL = torch.from_numpy(imgL).unsqueeze(0)
        imgR = torch.from_numpy(imgR).unsqueeze(0)

        return imgL, imgR
    """

    def estimate(self, left_image, right_image):
        left_img, right_img = self.preprocess(left_image, right_image)
        self.model.eval()
        outputs = self.model(left_img.to(self.device), right_img.to(self.device))

        # Extract the proposed disparity pyramid from outputs. The model returns a dict
        # with key 'prop_disp_pyramid' during eval, and that value is typically a list
        # of tensors (different pyramid levels). We select the highest-resolution / first
        # pyramid entry and convert it to a NumPy 2D disparity map.
        if isinstance(outputs, dict):
            prop = outputs.get('prop_disp_pyramid', None)
        else:
            prop = outputs

        # If pyramid is a list, take the first (final) element
        if isinstance(prop, list) and len(prop) > 0:
            disp_t = prop[0]
        else:
            disp_t = prop

        # Convert tensor -> numpy disparity map (H, W)
        if torch.is_tensor(disp_t):
            disp_np = disp_t.detach().cpu().numpy()
            # handle shapes: (B,1,H,W) or (B,H,W) or (H,W)
            if disp_np.ndim == 4:
                disp_np = disp_np[0, 0, :, :]
            elif disp_np.ndim == 3:
                # assume (B,H,W)
                disp_np = disp_np[0, :, :]
            elif disp_np.ndim == 2:
                pass
            else:
                # unexpected shape; try to squeeze
                disp_np = np.squeeze(disp_np)

            print(f"prop_disp_pyramid -> disparity map shape: {disp_np.shape}")
            return disp_np

        # If we get here, return prop as best-effort (may be list or other)
        return prop

if __name__ == "__main__":
    estimator = HitNetEstimator()
    print("estimator: {}".format(estimator))

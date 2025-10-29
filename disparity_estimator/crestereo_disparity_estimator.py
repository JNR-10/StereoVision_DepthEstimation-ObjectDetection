import torch
import torch.nn.functional as F
import numpy as np
import cv2

from flopth import flopth
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchstat import stat

import config
from networks.CREStereo.nets import CREStereo

class CREStereoEstimator:
    def __init__(self):
        # build model
        self.model = CREStereo(max_disp=256, mixed_precision=False, test_mode=True)

        # decide device safely (prefer DEVICE1 if present)
        desired = getattr(config, 'DEVICE1', getattr(config, 'DEVICE', 'cpu'))
        desired_str = str(desired)
        if 'cuda' in desired_str and torch.cuda.is_available():
            self.device = torch.device(desired_str)
        else:
            self.device = torch.device('cpu')
            if 'cuda' in desired_str:
                print(f"Warning: config.DEVICE1='{desired}' requests CUDA but CUDA is not available; using CPU instead.")

        # load checkpoint robustly
        try:
            state = torch.load(config.CRESTEREO_MODEL_PATH)
        except Exception as e:
            print(f"Warning: failed to load CREStereo checkpoint normally ({e}). Retrying with map_location='cpu'.")
            state = torch.load(config.CRESTEREO_MODEL_PATH, map_location=torch.device('cpu'))

        sd = state.get('model', state.get('state_dict', state))
        if isinstance(sd, dict):
            new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        else:
            new_sd = sd

        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.n_iter = 20
    def profile(self):
        print("Profiling Architecture : {}".format(config.ARCHITECTURE))
        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))
        model = CREStereo(max_disp=256, mixed_precision=False, test_mode=True)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))

        dummy_inputs = torch.rand(1, 3, width, height)
        print("=====START Profile With FLOPTH========")
        flops, params = flopth(model, inputs=(dummy_inputs,dummy_inputs))
        print("With flopth -> FLOPS: {}, params: {}".format(flops, params))
        print("=====END Profile With FLOPTH========")

    def load_image(self, image):
        return cv2.imread(image)
    def preprocess(self, left_image, right_image):
        left_img = left_image #cv2.imread(left_image)
        right_img = right_image #cv2.imread(right_image)
        w, h = left_img.shape[:2]
        crop_w, crop_h = 512, 256

        x1 = 50  # random.randint(0, w - crop_w)
        y1 = 50  # random.randint(0, h - crop_h)

        # random crop
        left_img = left_img[y1:y1 + crop_h, x1:x1 + crop_w]  # .crop((x1, y1, x1 + crop_w, y1 + crop_h))
        right_img = right_img[y1:y1 + crop_h, x1:x1 + crop_w]  # .crop((x1, y1, x1 + crop_w, y1 + crop_h))

        in_h, in_w = left_img.shape[:2]
        print("in_h: {}".format(in_h))
        print("in_w: {}".format(in_w))
        # Resize image in case the GPU memory overflows
        eval_h, eval_w = (in_h, in_w)
        assert eval_h % 8 == 0, "input height should be divisible by 8"
        assert eval_w % 8 == 0, "input width should be divisible by 8"

        imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

        return imgL, imgR, in_w, in_h
    def estimate(self, left_image, right_image):
        left = self.load_image(left_image)
        right = self.load_image(right_image)

        in_h, in_w = left.shape[:2]
        print("in_h: {}".format(in_h))
        print("in_w: {}".format(in_w))

        left_img, right_img, _, _ = self.preprocess(left, right)
        imgL = left_img.transpose(2, 0, 1)
        imgR = right_img.transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL[None, :, :, :])
        imgR = np.ascontiguousarray(imgR[None, :, :, :])

        imgL = torch.tensor(imgL.astype("float32")).to(self.device)
        imgR = torch.tensor(imgR.astype("float32")).to(self.device)

        imgL_dw2 = F.interpolate(
            imgL,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        imgR_dw2 = F.interpolate(
            imgR,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        # print(imgR_dw2.shape)
        with torch.inference_mode():
            pred_flow_dw2 = self.model(imgL_dw2, imgR_dw2, iters=self.n_iter, flow_init=None)

            pred_flow = self.model(imgL, imgR, iters=self.n_iter, flow_init=pred_flow_dw2)
        pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()
        t = float(in_w) / float(in_w)
        disp = cv2.resize(pred_disp, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

        return disp

if __name__ == "__main__":
    model = CREStereo(max_disp=256, mixed_precision=False, test_mode=True)
    print(model)


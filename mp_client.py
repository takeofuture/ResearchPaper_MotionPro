#!/usr/bin/env python3
# motionpro_cli.py  ← 好きな名前で保存

import os, json, uuid, argparse, warnings
import numpy as np, torch, cv2
from PIL import Image
from einops import rearrange, repeat
from scipy.interpolate import PchipInterpolator
from torchvision import transforms
from vtdm.utils import (import_filename, file2data, adaptively_load_state_dict,
                        ensure_dirname, data2file)
warnings.filterwarnings("ignore")

# --- 設定（Gradio デモと同じ値） ------------------------------
CKPT_PATH  = "models/motionpro_ckpt/MotionPro-gs_16k.pt"
CFG_PATH   = "vtdm/motionpro_net.py"
DEVICE     = "cuda:0"         # CPU なら "cpu"
MODEL_LEN  = 16
BUCKET_ID  = 17
# -------------------------------------------------------------

# -------------------------------------------------------------
# 1)  demo_sparse_flex_wh.py から Drag クラスを丸ごと移植
#     （Gradio は import しない）
# -------------------------------------------------------------
class Drag:
    def __init__(self, device, model_path, cfg_path, model_length):
        self.device        = device
        self.model_length  = model_length

        # --- ネットワークを動的 import ------------------------
        cf         = import_filename(cfg_path)          # 一回だけ
        Net, args  = cf.Net, cf.args
        #self.net   = Net(args).to(device).eval()
        #self.motionpro_net = Net(args).to(device).eval()
        motionpro_net = Net(args)   # ① インスタンス化
        motionpro_net.to(device)    # ② デバイス転送（戻り値を無視）
        motionpro_net.eval()        # ③ eval モードに
        self.motionpro_net = motionpro_net
        # --- checkpoint ロード (.pt/.pth も .ckpt も対応) -----
        if model_path.endswith(("pt", "pth")):
            sd_raw = torch.load(model_path, map_location="cpu")
            state  = {k.removeprefix("module."): v
                      for k, v in sd_raw["module"].items()}
        else:                                           # .ckpt
            state  = torch.load(model_path, map_location="cpu")
            state  = state["state_dict"] if "state_dict" in state else state

        #_miss, _unexp = self.net.load_state_dict(state, strict=False)
        _miss, _unexp = self.motionpro_net.load_state_dict(state, strict=False)
        print(f"[CKPT] loaded {model_path}")

    # -------- Gradio デモと同じ推論ルーチン ------------------
    @torch.no_grad()
    def forward_sample(self, drag_t, first_frame, bucket_id):
        b,l,h,w,c   = drag_t.shape
        flow        = rearrange(drag_t, 'b l h w c -> b l c h w')
        cond = {
            "cond_frames_without_noise": first_frame,
            "cond_frames": first_frame + 0.02*torch.randn_like(first_frame),
            "motion_bucket_id": torch.full((b*l,), bucket_id,
                                           device=self.device, dtype=torch.long),
            "fps_id":           torch.full((b*l,), self.motionpro_net.args.fps,
                                           device=self.device, dtype=torch.long),
            "cond_aug":         torch.full((b*l,), 0.02,
                                           device=self.device)
        }
        # uc/c のブロードキャスト
        uc = {k:v.clone() for k,v in cond.items() if isinstance(v,torch.Tensor)}
        #c, uc = self.net.conditioner.get_unconditional_conditioning(
        c, uc = self.motionpro_net.conditioner.get_unconditional_conditioning(
                    cond, batch_uc=uc,
                    force_uc_zero_embeddings=["cond_frames",
                                              "cond_frames_without_noise"])
        for k in ["crossattn","concat"]:
            c[k]  = rearrange(repeat(c[k],  'b ... -> b t ...',
                                     t=self.motionpro_net.num_frames),
                              'b t ... -> (b t) ...')
            uc[k] = rearrange(repeat(uc[k], 'b ... -> b t ...',
                                     t=self.motionpro_net.num_frames),
                              'b t ... -> (b t) ...')

        #shape = (self.net.num_frames,4,h//8,w//8)
        shape = (self.motionpro_net.num_frames,4,h//8,w//8)
        noise = torch.randn(shape, device=self.device)
        extra = {
            "image_only_indicator": torch.zeros(2, self.motionpro_net.num_frames,
                                                device=self.device),
            "num_video_frames": self.motionpro_net.num_frames,
            "flow": flow.repeat(2,1,1,1,1)
        }
        def denoiser(x,s,c_):   # sampler が要求するコールバック
            #return self.net.denoiser(self.net.model, x, s, c_, **extra)
            return self.motionpro_net.denoiser(self.motionpro_net.model,
                                               x, s, c_, **extra)
        #z   = self.net.sampler(denoiser, noise, cond=c, uc=uc)
        #vid = self.net.decode_first_stage(z)              # (b*t,3,H,W)
        z   = self.motionpro_net.sampler(denoiser, noise, cond=c, uc=uc)
        vid = self.motionpro_net.decode_first_stage(z)
        return rearrange(vid, '(b t) c h w -> b t c h w', b=b)[0]
# -------------------------------------------------------------

# 2) drag/mask テンソルを組む補助関数 --------------------------
def _interp_track(tr):
    if len(tr)==1: tr = [tr[0], (tr[0][0]+1, tr[0][1]+1)]
    x,y = zip(*tr); t=np.linspace(0,1,len(tr))
    fx,fy = PchipInterpolator(t,x), PchipInterpolator(t,y)
    xs,ys = fx(np.linspace(0,1,MODEL_LEN)), fy(np.linspace(0,1,MODEL_LEN))
    return list(zip(xs,ys))

def build_drag_mask(tracks, H, W, mask_rgb):
    drag  = torch.zeros(MODEL_LEN, H, W, 2)
    mask0 = torch.zeros(MODEL_LEN, H, W, 1)

    for tr in tracks:
        tr=_interp_track(tr)
        x0,y0 = int(tr[0][0]), int(tr[0][1])
        for i,(x,y) in enumerate(tr):
            drag[i,y0-4:y0+4,x0-4:x0+4,0] = x-x0
            drag[i,y0-4:y0+4,x0-4:x0+4,1] = y-y0
            mask0[i,y0-4:y0+4,x0-4:x0+4] = 1

    brush = (np.array(mask_rgb)[:,:,0]>0).astype(np.uint8)
    brush = torch.from_numpy(brush)[None,...,None].repeat(MODEL_LEN,1,1,1)
    mask  = torch.logical_or(mask0.bool(), brush.bool()).float()

    return torch.cat([drag,mask], -1)[None]      # (1,L,H,W,3)

# 3) 画像 → GIF 変換補助
def tensor2pil(t):
    arr=(t.permute(1,2,0).cpu().numpy()*127.5+127.5).clip(0,255).astype('uint8')
    return Image.fromarray(arr)

# -------------------- main ---------------------------------
def main(img, mask, traj, out, gifname=None):
    ensure_dirname(out)
    img_pil  = Image.open(img).convert("RGB")
    mask_pil = Image.open(mask).convert("RGB")
    H,W      = img_pil.height, img_pil.width

    tracks   = json.load(open(traj))
    drag_t   = build_drag_mask(tracks, H, W, mask_pil).to(DEVICE)
    first    = transforms.ToTensor()(img_pil)[None].to(DEVICE)*2 - 1

    runner   = Drag(DEVICE, CKPT_PATH, CFG_PATH, MODEL_LEN)
    vid      = runner.forward_sample(drag_t, first, BUCKET_ID)   # (T,3,H,W)

    frames   = [tensor2pil(f) for f in vid]
    gif_path = os.path.join(out, f"motionpro_{uuid.uuid4().hex}.gif")
    data2file(frames, gif_path, printable=False, duration=1/8, override=True)
    # gifname が指定されていればリネーム
    if gifname:
        final_path = os.path.join(out, gifname)
        # 親ディレクトリがなければ作成
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        os.replace(gif_path, final_path)
    else:
        final_path = gif_path
    print("✔ GIF saved:", final_path)
    
# CLI -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",  required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--traj", required=True)
    parser.add_argument("--out",  default="output_cli")
    parser.add_argument("--gifname", default=None, help="（任意）最終GIFのファイル名")
    args = parser.parse_args()
    main(args.img, args.mask, args.traj, args.out, args.gifname)

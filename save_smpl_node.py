from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np
import pickle

from hmr4d.utils.pylogger import Log


class SaveSMPL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS",),
                "npz_output_path": (
                    "STRING",
                    {
                        "default": "output/motion.npz",
                        "multiline": False,
                    },
                ),
                "pkl_output_path": (
                    "STRING",
                    {
                        "default": "output_pkl/motion.pkl",
                        "multiline": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "info")
    FUNCTION = "save_smpl"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/SMPL"

    def save_smpl(
        self,
        smpl_params: Dict,
        npz_output_path: str,
        pkl_output_path: str,
    ) -> Tuple[str, str]:
        try:
            Log.info("[SaveSMPL] Saving SMPL motion data (NPZ + PKL)...")

            if "global" not in smpl_params:
                raise ValueError(
                    "smpl_params に 'global' が含まれていません。"
                )

            global_params = smpl_params["global"]

            npz_path = Path(npz_output_path)
            pkl_path = Path(pkl_output_path)

            npz_path.parent.mkdir(parents=True, exist_ok=True)
            pkl_path.parent.mkdir(parents=True, exist_ok=True)

            if npz_path.suffix == "":
                npz_path = npz_path.with_suffix(".npz")
            if pkl_path.suffix == "":
                pkl_path = pkl_path.with_suffix(".pkl")

            np_params: Dict[str, np.ndarray] = {}
            for key, value in global_params.items():
                if isinstance(value, torch.Tensor):
                    np_params[key] = value.detach().cpu().numpy()
                else:
                    np_params[key] = np.array(value)

            np.savez(npz_path, **np_params)

            pred_np: Dict[str, Dict[str, np.ndarray]] = {
                "smpl_params_global": {},
                "smpl_params_incam": {},
            }

            required_keys = ["body_pose", "global_orient", "transl", "betas"]
            for key in required_keys:
                if key in np_params:
                    arr = np_params[key]
                else:
                    if key not in global_params:
                        raise KeyError(f"'global' に '{key}' が含まれていません。")
                    val = global_params[key]
                    if isinstance(val, torch.Tensor):
                        arr = val.detach().cpu().numpy()
                    else:
                        arr = np.array(val)

                pred_np["smpl_params_global"][key] = arr
                pred_np["smpl_params_incam"][key] = arr

            with open(pkl_path, "wb") as handle:
                pickle.dump(pred_np, handle, protocol=pickle.HIGHEST_PROTOCOL)

            num_frames = pred_np["smpl_params_global"]["body_pose"].shape[0]
            file_size_kb = npz_path.stat().st_size / 1024
            file_size_pkl_kb = pkl_path.stat().st_size / 1024

            info = (
                f"SaveSMPL Complete\n"
                f"NPZ: {npz_path} ({file_size_kb:.1f} KB)\n"
                f"PKL: {pkl_path} ({file_size_pkl_kb:.1f} KB)\n"
                f"Frames: {num_frames}\n"
            )

            Log.info(
                f"[SaveSMPL] Saved {num_frames} frames to {npz_path} and {pkl_path}"
            )
            return (str(npz_path.absolute()), info)

        except Exception as e:
            error_msg = f"SaveSMPL failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ("", error_msg)


NODE_CLASS_MAPPINGS = {
    "SaveSMPL": SaveSMPL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveSMPL": "Save SMPL Motion",
}

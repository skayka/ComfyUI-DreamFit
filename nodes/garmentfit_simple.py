"""
Simplified GarmentFit nodes for ComfyUI
Minimal dependencies version
"""

class GarmentFitLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint_name": (["flux_i2i.bin"], ),
            }
        }
    
    RETURN_TYPES = ("GARMENTFIT",)
    FUNCTION = "load_model"
    CATEGORY = "dreamfit"
    
    def load_model(self, checkpoint_name):
        # Simple placeholder that just returns a dict
        return ({"checkpoint": checkpoint_name, "loaded": True},)


class GarmentFitApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
                "garmentfit": ("GARMENTFIT", ),
                "garment_image": ("IMAGE", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "dreamfit"
    
    def apply(self, model, clip, garmentfit, garment_image, positive, negative, strength):
        # For now, just return the model unchanged
        # This proves the nodes load
        print(f"[GarmentFit] Would apply with strength {strength}")
        return (model,)


NODE_CLASS_MAPPINGS = {
    "GarmentFitLoader": GarmentFitLoader,
    "GarmentFitApply": GarmentFitApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarmentFitLoader": "Load GarmentFit (Simple)",
    "GarmentFitApply": "Apply GarmentFit (Simple)",
}
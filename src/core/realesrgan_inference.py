import argparse
import cv2
import os
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def load_model(model_path: str, scale: int = 4, device: str = 'cuda'):
    # Étape 1 : Créer l'architecture
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,  # Change to 6 for anime model
        num_grow_ch=32,
        scale=scale
    )
    # Étape 2 : Wrapper avec RealESRGANer
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,        # Pas de tiling pour commencer
        tile_pad=10,
        pre_pad=0,
        half=False,    # Utilisez FP32 pour commencer
        device=device
    )
    return upsampler

def upscale_image(upsampler, input_path: str, output_path: str, outscale: int = 4):
    # Lire
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {input_path}")
    print(f"Image source : {img.shape}")  # Afficher la taille
    # Upscaler
    output, _ = upsampler.enhance(img, outscale=outscale)
    print(f"Image upscalée : {output.shape}")
    # Sauvegarder
    cv2.imwrite(output_path, output)
    print(f"Image sauvegardée : {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Upscale images using Real-ESRGAN')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output image')
    parser.add_argument('--model_path', '-m', type=str, required=True,
                       help='Path to model weights (.pth)')
    parser.add_argument('--scale', '-s', type=int, default=4,
                       help='Upscaling factor (default: 4)')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    args = parser.parse_args()
    # Charger le modèle
    print(f"Loading model from {args.model_path}...")
    upsampler = load_model(args.model_path, args.scale, args.device)
    # Upscaler l'image
    print(f"Upscaling {args.input}...")
    upscale_image(upsampler, args.input, args.output, args.scale)
    print("Done!")

if __name__ == '__main__':
    main()
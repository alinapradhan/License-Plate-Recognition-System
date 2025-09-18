"""
Export trained models to various formats for deployment.
"""
import torch
import onnx
from pathlib import Path
from ultralytics import YOLO
import argparse


def export_to_onnx(model_path: str, output_path: str = None, img_size: int = 640):
    """
    Export YOLOv8 model to ONNX format.
    
    Args:
        model_path: Path to trained YOLOv8 model
        output_path: Output path for ONNX model
        img_size: Input image size
    
    Returns:
        Path to exported ONNX model
    """
    try:
        print(f"Loading YOLOv8 model from: {model_path}")
        model = YOLO(model_path)
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = f"{model_name}.onnx"
        
        print(f"Exporting to ONNX format...")
        exported_path = model.export(
            format='onnx',
            imgsz=img_size,
            optimize=True,
            simplify=True
        )
        
        print(f"✓ ONNX model exported to: {exported_path}")
        
        # Verify the exported model
        onnx_model = onnx.load(exported_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
        
        return exported_path
        
    except Exception as e:
        print(f"✗ Failed to export to ONNX: {e}")
        return None


def export_to_tflite(model_path: str, output_path: str = None, img_size: int = 640):
    """
    Export YOLOv8 model to TensorFlow Lite format.
    
    Args:
        model_path: Path to trained YOLOv8 model
        output_path: Output path for TFLite model
        img_size: Input image size
    
    Returns:
        Path to exported TFLite model
    """
    try:
        print(f"Loading YOLOv8 model from: {model_path}")
        model = YOLO(model_path)
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = f"{model_name}.tflite"
        
        print(f"Exporting to TensorFlow Lite format...")
        exported_path = model.export(
            format='tflite',
            imgsz=img_size
        )
        
        print(f"✓ TensorFlow Lite model exported to: {exported_path}")
        return exported_path
        
    except Exception as e:
        print(f"✗ Failed to export to TensorFlow Lite: {e}")
        return None


def export_to_torchscript(model_path: str, output_path: str = None, img_size: int = 640):
    """
    Export YOLOv8 model to TorchScript format.
    
    Args:
        model_path: Path to trained YOLOv8 model
        output_path: Output path for TorchScript model
        img_size: Input image size
    
    Returns:
        Path to exported TorchScript model
    """
    try:
        print(f"Loading YOLOv8 model from: {model_path}")
        model = YOLO(model_path)
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = f"{model_name}.torchscript"
        
        print(f"Exporting to TorchScript format...")
        exported_path = model.export(
            format='torchscript',
            imgsz=img_size
        )
        
        print(f"✓ TorchScript model exported to: {exported_path}")
        return exported_path
        
    except Exception as e:
        print(f"✗ Failed to export to TorchScript: {e}")
        return None


def export_to_coreml(model_path: str, output_path: str = None, img_size: int = 640):
    """
    Export YOLOv8 model to Core ML format (for iOS deployment).
    
    Args:
        model_path: Path to trained YOLOv8 model
        output_path: Output path for Core ML model
        img_size: Input image size
    
    Returns:
        Path to exported Core ML model
    """
    try:
        print(f"Loading YOLOv8 model from: {model_path}")
        model = YOLO(model_path)
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = f"{model_name}.mlpackage"
        
        print(f"Exporting to Core ML format...")
        exported_path = model.export(
            format='coreml',
            imgsz=img_size
        )
        
        print(f"✓ Core ML model exported to: {exported_path}")
        return exported_path
        
    except Exception as e:
        print(f"✗ Failed to export to Core ML: {e}")
        return None


def export_all_formats(model_path: str, output_dir: str = "exported_models", img_size: int = 640):
    """
    Export model to all supported formats.
    
    Args:
        model_path: Path to trained YOLOv8 model
        output_dir: Directory to save exported models
        img_size: Input image size
    
    Returns:
        Dictionary of exported model paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_name = Path(model_path).stem
    exported_models = {}
    
    print(f"Exporting {model_name} to multiple formats...")
    
    # ONNX (cross-platform)
    onnx_path = output_path / f"{model_name}.onnx"
    exported_models['onnx'] = export_to_onnx(model_path, str(onnx_path), img_size)
    
    # TensorFlow Lite (mobile)
    tflite_path = output_path / f"{model_name}.tflite"
    exported_models['tflite'] = export_to_tflite(model_path, str(tflite_path), img_size)
    
    # TorchScript (PyTorch deployment)
    torchscript_path = output_path / f"{model_name}.torchscript"
    exported_models['torchscript'] = export_to_torchscript(model_path, str(torchscript_path), img_size)
    
    # Core ML (iOS) - optional
    try:
        coreml_path = output_path / f"{model_name}.mlpackage"
        exported_models['coreml'] = export_to_coreml(model_path, str(coreml_path), img_size)
    except Exception as e:
        print(f"Core ML export skipped: {e}")
    
    return {k: v for k, v in exported_models.items() if v is not None}


def optimize_for_inference(onnx_path: str, output_path: str = None):
    """
    Optimize ONNX model for faster inference.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path for optimized model
    
    Returns:
        Path to optimized model
    """
    try:
        import onnxruntime as ort
        
        if output_path is None:
            base_path = Path(onnx_path)
            output_path = base_path.parent / f"{base_path.stem}_optimized.onnx"
        
        print(f"Optimizing ONNX model for inference...")
        
        # Create optimization session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = str(output_path)
        
        # Run optimization
        session = ort.InferenceSession(onnx_path, sess_options)
        
        print(f"✓ Optimized ONNX model saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"✗ Failed to optimize ONNX model: {e}")
        return onnx_path


def create_deployment_config(exported_models: dict, output_path: str = "deployment_config.yaml"):
    """
    Create deployment configuration file.
    
    Args:
        exported_models: Dictionary of exported model paths
        output_path: Path to save configuration
    """
    config = {
        'models': {},
        'deployment': {
            'input_size': [640, 640],
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'class_names': ['license_plate']
        },
        'formats': {}
    }
    
    for format_name, model_path in exported_models.items():
        if model_path:
            config['models'][format_name] = str(model_path)
            
            # Add format-specific configurations
            if format_name == 'onnx':
                config['formats']['onnx'] = {
                    'providers': ['CPUExecutionProvider', 'CUDAExecutionProvider'],
                    'optimization': True
                }
            elif format_name == 'tflite':
                config['formats']['tflite'] = {
                    'num_threads': 4,
                    'use_gpu': False
                }
            elif format_name == 'torchscript':
                config['formats']['torchscript'] = {
                    'device': 'auto'
                }
    
    # Save configuration
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✓ Deployment configuration saved to: {output_path}")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to multiple formats")
    
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 model")
    parser.add_argument("--output-dir", default="exported_models", help="Output directory")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--format", choices=['onnx', 'tflite', 'torchscript', 'coreml', 'all'],
                       default='all', help="Export format")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"✗ Model file not found: {args.model}")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Export models
    if args.format == 'all':
        exported_models = export_all_formats(args.model, args.output_dir, args.img_size)
    else:
        model_name = Path(args.model).stem
        output_path = Path(args.output_dir) / f"{model_name}.{args.format}"
        
        exported_models = {}
        if args.format == 'onnx':
            exported_models['onnx'] = export_to_onnx(args.model, str(output_path), args.img_size)
        elif args.format == 'tflite':
            exported_models['tflite'] = export_to_tflite(args.model, str(output_path), args.img_size)
        elif args.format == 'torchscript':
            exported_models['torchscript'] = export_to_torchscript(args.model, str(output_path), args.img_size)
        elif args.format == 'coreml':
            exported_models['coreml'] = export_to_coreml(args.model, str(output_path), args.img_size)
    
    # Optimize ONNX model if requested
    if args.optimize and 'onnx' in exported_models and exported_models['onnx']:
        optimized_path = optimize_for_inference(exported_models['onnx'])
        exported_models['onnx_optimized'] = optimized_path
    
    # Create deployment configuration
    config_path = Path(args.output_dir) / "deployment_config.yaml"
    create_deployment_config(exported_models, str(config_path))
    
    # Summary
    print("\n🎉 Export Summary:")
    for format_name, model_path in exported_models.items():
        if model_path:
            print(f"  ✓ {format_name.upper()}: {model_path}")
        else:
            print(f"  ✗ {format_name.upper()}: Failed to export")
    
    print(f"\nDeployment configuration: {config_path}")


if __name__ == "__main__":
    main()
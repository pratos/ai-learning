import torch
import typer

from src.encoding_101.models.autoencoder_annotated import NVTXVanillaAutoencoder
from src.encoding_101.training.trainer import train_autoencoder

app = typer.Typer(
    help="Profile training loop with NVTX annotations",
    add_completion=False
)


@app.command()
def profile(
    data_dir: str = typer.Option("./data", help="Directory where the data will be stored"),
    latent_dim: int = typer.Option(128, help="Dimensionality of the latent space"),
    batch_size: int = typer.Option(64, help="Batch size for training and validation"),
    max_epochs: int = typer.Option(3, help="Maximum number of epochs (short for profiling)"),
    device_id: int = typer.Option(0, help="GPU device ID to use"),
    enable_nvtx: bool = typer.Option(True, help="Enable NVTX annotations"),
    profile_first_n_batches: int = typer.Option(0, help="If > 0, only profile first N batches per epoch"),
    disable_mar_viz: bool = typer.Option(True, help="Disable MAR visualization for cleaner profiling"),
    num_workers: int = typer.Option(0, help="Number of DataLoader workers (0=safe, 2-4=faster but may cause segfaults)"),
):
    """Run training with NVTX profiling enabled for performance analysis"""
    
    # Print profiling information
    typer.echo("🔍 NVTX Training Profiler")
    typer.echo("=" * 50)
    typer.echo(f"NVTX annotations: {'✅ enabled' if enable_nvtx else '❌ disabled'}")
    typer.echo(f"Epochs: {max_epochs} (shortened for profiling)")
    typer.echo(f"Batch size: {batch_size}")
    typer.echo(f"Device: GPU {device_id}")
    typer.echo(f"DataLoader workers: {num_workers} ({'⚠️  may cause segfaults' if num_workers > 0 else '✅ stable'})")
    typer.echo(f"MAR visualization: {'❌ disabled' if disable_mar_viz else '✅ enabled'}")
    
    if profile_first_n_batches > 0:
        typer.echo(f"Profiling first {profile_first_n_batches} batches per epoch only")
    
    typer.echo()
    
    # Check NVTX availability
    try:
        import nvtx
        typer.echo("✅ NVTX package available")
    except ImportError:
        typer.echo("❌ NVTX package not found. Install with: pip install nvtx")
        return
    
    # Check CUDA availability
    import torch
    if torch.cuda.is_available():
        typer.echo(f"✅ CUDA available: {torch.cuda.get_device_name(device_id)}")
    else:
        typer.echo("❌ CUDA not available")
        return
    
    typer.echo()
    
    # Setup model configuration
    model_kwargs = {
        "latent_dim": latent_dim,
        "visualize_mar": not disable_mar_viz,
        "mar_viz_epochs": max_epochs + 1 if disable_mar_viz else 2,  # Disable or reduce frequency
        "mar_samples_per_class": 3,  # Fewer samples for faster profiling
        "enable_nvtx": enable_nvtx,
    }
    
    # Run training
    try:
        typer.echo("🚀 Starting profiling run...")
        
        model, trainer = train_autoencoder(
            model_class=NVTXVanillaAutoencoder,
            model_kwargs=model_kwargs,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device_id=device_id,
            max_epochs=max_epochs,
            debug=False,
        )
        
        typer.echo("✅ Profiling run complete!")
        typer.echo()
        typer.echo("📊 Analysis Instructions:")
        typer.echo("1. Install Nsight Systems: https://developer.nvidia.com/nsight-systems")
        typer.echo("2. Open the generated .qdrep file with: nsys-ui <filename>.qdrep")
        typer.echo("3. Look for NVTX annotations in the timeline view")
        typer.echo("4. Analyze GPU utilization and identify bottlenecks")
        
    except KeyboardInterrupt:
        typer.echo("❌ Profiling interrupted by user")
    except Exception as e:
        typer.echo(f"❌ Error during profiling: {e}")
        raise


@app.command()
def test_nvtx():
    """Test NVTX functionality without running full training"""
    
    typer.echo("🧪 Testing NVTX functionality...")
    
    try:
        import nvtx
        import torch
        import time
        
        typer.echo("✅ NVTX package imported successfully")
        
        # Test basic annotation
        with nvtx.annotate("Test Annotation", color="blue"):
            time.sleep(0.1)
        
        typer.echo("✅ Basic NVTX annotation test passed")
        
        # Test GPU annotation if available
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            typer.echo(f"✅ CUDA device: {torch.cuda.get_device_name(device)}")
            
            with nvtx.annotate("GPU Test", color="green"):
                x = torch.randn(1000, 1000).cuda()
                y = torch.matmul(x, x.T)
                torch.cuda.synchronize()
            
            typer.echo("✅ GPU NVTX annotation test passed")
        else:
            typer.echo("⚠️  CUDA not available, GPU tests skipped")
        
        typer.echo("🎉 All NVTX tests passed!")
        typer.echo()
        typer.echo("You can now run profiling with:")
        typer.echo("nsys profile --trace=nvtx,cuda python scripts/profile_training.py profile")
        
    except ImportError as e:
        typer.echo(f"❌ Import error: {e}")
        typer.echo("Install NVTX with: pip install nvtx")
    except Exception as e:
        typer.echo(f"❌ Test failed: {e}")

if __name__ == "__main__":
    app() 
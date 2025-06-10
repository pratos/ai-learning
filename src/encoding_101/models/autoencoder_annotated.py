from src.encoding_101.mixins.nvtx import NVTXProfilingMixin
from src.encoding_101.models.cnn_autoencoder import CNNAutoencoder
from src.encoding_101.models.vanilla_autoencoder import VanillaAutoencoder


class NVTXVanillaAutoencoder(NVTXProfilingMixin, VanillaAutoencoder):
    """Vanilla Autoencoder with comprehensive NVTX profiling annotations"""

    def __init__(
        self,
        latent_dim: int = 128,
        visualize_mar: bool = False,
        mar_viz_epochs: int = 5,
        mar_samples_per_class: int = 5,
        enable_nvtx: bool = True,
    ):
        super().__init__(
            latent_dim=latent_dim,
            visualize_mar=visualize_mar,
            mar_viz_epochs=mar_viz_epochs,
            mar_samples_per_class=mar_samples_per_class,
            enable_nvtx=enable_nvtx,
        )

class NVTXCNNAutoencoder(NVTXProfilingMixin, CNNAutoencoder):
    """CNN Autoencoder with comprehensive NVTX profiling annotations"""

    def __init__(
        self,
        latent_dim: int = 128,
        visualize_mar: bool = False,
        mar_viz_epochs: int = 5,
        mar_samples_per_class: int = 5,
        enable_nvtx: bool = True,
    ):
        super().__init__(
            latent_dim=latent_dim,
            visualize_mar=visualize_mar,
            mar_viz_epochs=mar_viz_epochs,
            mar_samples_per_class=mar_samples_per_class,
            enable_nvtx=enable_nvtx,
        )
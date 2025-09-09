# Media generation

### VAEs

The weights are stored in FP16 due to size limitations on GitHub - halved from FP32 which is the default. The main script
has a `short_run` option to test out pipeline changes by running for one epoch.

Result after training on "flwrlabs/celeba" from HuggingFace for 20 epochs: 
![generated_faces](vae/samples/faces_vae_E020_I002_D20250909-181649_N16.png)


For comparison, here are the samples from a run after just 1 epoch: ![1_epoch_output](/vae/samples/faces_vae_E001_I001_D20250908-232617_N16.png)

Clearly needs more training

### GANs (...)

### Diffusion (...)
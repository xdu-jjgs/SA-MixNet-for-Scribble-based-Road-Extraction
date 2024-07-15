# SA-MixNet
1) First, pip install `opencv-python == 4.6.0.66` and replace the `\anaconda3\envs\pytorch\Lib\site-packages\skimage\segmentation\slic_superpixels.py` by the `slic_superpixels.py` in this repository,
to get Road Seed Guided SLIC.
2) Run `road_label_propagation.py` to generate proposal masks.
3) Run `train_mix_D.py` for training and run `test.py` for testing.



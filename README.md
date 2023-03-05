# YOLOv1

YOLOv1 from scratch in pytorch. Almost completely works but the non max supression probably needs a little tuning

# Training

Edit the `train.py` file to adjust batch size, optimizer parameters, and learning rate parameters then just run the file

# Inference

Just put model in eval mode with `model.eval()` and use `show_image((image, output))` to display image with bounding boxes.

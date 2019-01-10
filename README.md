# ResidualDetectron
Runs MaskR-CNN (Detectron) on residual images of a source separation algorithm to identify undetected sources.

The goal here is to identify previously undetected galaxies from the model of a given scene and it's residual.
We run a detection algorithm which is a modified version of an [implementation](https://github.com/matterport/Mask_RCNN)
of the [Mask R-CNN](https://arxiv.org/abs/1703.06870) architecture.
The project [webpage](https://sowmyakth.github.io/ResidualDetectron/) has a detailed explanation about the
nature of the problem, our analysis method and results.
Since we only want the position of undetected sources we don't generate segmentation maps for the detections.

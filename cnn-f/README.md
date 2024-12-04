# CNN-F Model Notes

This is a link to the [paper](https://arxiv.org/pdf/1912.11035) and [GitHub repository](https://github.com/PeterWang512/CNNDetection) associated with the CNN-F model. 

### Steps to run the model
As outlined in the ReadME file of the Github repository, I followed [Section 1: Setup](https://github.com/PeterWang512/CNNDetection?tab=readme-ov-file#1-setup) and [Section 2: Quick Start](https://github.com/PeterWang512/CNNDetection?tab=readme-ov-file#2-quick-start). This involved running the `demo.py` or `demo_dir.py` files depending on if you want to evaluate the model on single images or whole datasets.


### Challenges
There was one main challenge I ran into when running the CNN-F model. When running the model on our chosen datasets, there was a mismatch in the size of our dataset images and the size of the images that the model works with. To solve this issue, I wrote a script that resized all the dataset images to the specified 256x256 pixels. This allowed the model to run without issues. An alternative to resizing the images is to specify certain parameters involving whether or not to crop the images, and how to crop them (i.e., center-cropped versus cropping a random section of the image). Specifics regarding these parameters are outlined in the paper as well as in the `demo.py` or `demo_dir.py` files. 

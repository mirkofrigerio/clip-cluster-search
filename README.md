# Clustering of semantically similar products

### What are we trying to build?
We'd like to build a solution capable of returning the products that best match the semantics of the text input. We should be able to type "halloween" or "orange shirts" and get back a group of items visually related to those categories.

To do this, we need to have both images and text represented in the same embedding space. Then we can look at the closest images/text to our input text.

### Which model?
We need a text+image model. CLIP works perfectly for this. CLIP has seen millions of product images during pre-training, so fine-tuning with an Amazon products datasets is optional.



### Notes:
Possible datasets for fine-tuning:
- https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- https://github.com/Crossing-Minds/shopping-queries-image-dataset

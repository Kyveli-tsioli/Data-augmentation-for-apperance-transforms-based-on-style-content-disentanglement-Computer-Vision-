# MSc_project_DSML
The main contribution of this thesis is a novel, statistical, appearance-driven data augmentation method with multi-domain applicability. Our methodology is based on 
disentangling factors of variation in images, specifically content and style, where content is the component invariant to augmentation, while style is subject to 
change. Our approach accommodates the assumption that augmented samples maintain the semantic aspects (i.e. content) of the existing images in the dataset, while
exhibit variations in the overall image appearance. Specifically in our implementation, such changes among the original and the post-augmentation samples relate
to the overall image contrast and brightness. Our appearance-based data augmentation strategy effectively increases the diversity of samples in the training set.
Working with the Cityscapes dataset, we apply our data augmentation method to solve the segmentation task, i.e. the task of assigning a semantic label to each
pixel of an image \cite{zhang2020automatic}. We validate both quantitatively and qualitatively that our data augmentation approach slightly boosts segmentation
performance and yields more fine-grained object boundaries compared to the model where no appearance augmentation is performed.

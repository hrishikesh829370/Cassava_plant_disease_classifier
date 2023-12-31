# 🌿 Cassava Leaf Disease Identification Project 🌿
### Empowering Cassava Farmers with Disease Detection

## Project Overview

As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. However, viral diseases often lead to poor yields and threaten food security. Traditional methods of disease detection are labor-intensive and costly, relying on government-funded agricultural experts to visually inspect and diagnose cassava plants. These solutions suffer from limited accessibility in rural areas, where farmers may only have access to mobile-quality cameras with low bandwidth.

This project introduces a machine learning solution to quickly identify diseases in cassava leaves, providing a tool for farmers to safeguard their crops before irreparable damage occurs. The aim is to make the process efficient, accessible, and cost-effective.

## Dataset
![sample_images](https://github.com/hrishikesh829370/Cassava_plant_disease_classifier/assets/131910887/4234d58b-ecaa-4566-9294-26ae1a0c6fbf)

This project utilizes a dataset of 26,000 labeled cassava leaf images collected during surveys in Uganda. The dataset includes images sourced from farmers and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This dataset closely mirrors real-world conditions and the images farmers might need to diagnose.


## Disease Categories

Our project classifies cassava leaves into five categories:

1. Cassava Bacterial Blight (CBB)
2. Cassava Brown Streak Disease (CBSD)
3. Cassava Green Mottle (CGM)
4. Cassava Mosaic Disease (CMD)
5. Healthy

## Project Goals

The project aims to achieve the following objectives:

- Implement a machine learning model capable of accurately classifying cassava leaf images into their respective disease categories.
- Provide an easy-to-use tool for farmers to diagnose cassava diseases with minimal resources and accessibility constraints.
- Contribute to improving cassava crop yields and food security in Sub-Saharan Africa.

## Our Approach

### Dataset Preparation

The dataset is noisy and imbalanced, making it essential to address these issues. The following strategies has been employed:
**Train set**: ~26,000 images (a combination of 2020 and 2019 contest data).
**Test set**: ~15,000 images.

- Utilize Stratified K-folding to ensure equal class distribution in each fold.
- Implement robust loss functions, including Focal Loss, to handle the imbalanced dataset.

## To-do list

- [x] Code baseline and trainer on GPU + TPU  
- [x] Transforms: [albumentations](https://github.com/albumentations-team/albumentations)
- [x] Implement models: EfficientNet, ViT, Resnext 
- [x] Implement losses: Focal loss, CrossEntropy loss, Bi-Tempered Loss  
- [x] Implement optimizers: SAM  
- [x] Implement schedulers: StepLR, WarmupCosineSchedule  
- [x] Implement metrics: accuracy
- [x] Write inference notebook  
- [x] Implement Stratified K-folding  
- [x] Merge 2019 dataset and 2020 dataset from kaggle 
- [x] Implement gradient_accumulation   
- [x] Implement Automatic Mixed Precision  
- [x] Write Optuna scripts for hyperparams tuning  

##Summary
### Model Selection

This final model is a stacked ensemble of three CNN-based models: EfficientNet, ResneXt, and Densenet. This ensemble approach enhances classification accuracy.

![final_model-2](https://github.com/hrishikesh829370/Cassava_plant_disease_classifier/assets/131910887/fb4e96d9-a844-4cc4-a2ee-aeb53b323caa)



### Training configuration

-The dataset is, in fact, noisy (contains irrelevant images of the tree roots or distorted images) and clearly imbalanced.  

-Tackling the first problem by splitting the training set into 5 equal-size folds, while each fold has the same class distribution as the original set (this splitting scheme is called [Stratified K-folding](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)). This way of training gives every image in the training set a chance to contribute to the final predictions. 

-Tried to adapt some robust loss functions like [Bi-Tempered Loss](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html) and [Taylor CE Loss](https://www.ijcai.org/Proceedings/2020/0305.pdf), 


Most models used are from the [Timm library](https://github.com/rwightman/pytorch-image-models).

* resnext50_32x4d - [training config](src/configs/resnext.yaml)

* efficientnet-b4 - [training config](src/configs/effnet.yaml)

* densenet121 - [training config](src/configs/densenet.yaml)

**Result**

| Model          | fold0    | fold8    | fold2    | fold3    | fold4    | CV       | 
|----------------|----------|----------|----------|----------|----------|----------| 
| densenet121    | 0.884346 | 0.881308 | 0.878710 | 0.873802 | 0.888993 | 0.881431 | 
| effnetB4       | 0.889018 | 0.889252 | 0.881046 | 0.876840 | 0.888525 | 0.884936 | 
| resnext50_32x4 | 0.884813 | 0.880140 | 0.881748 | 0.878008 | 0.892498 | 0.883441 | 


# Hyperparameters search using Optuna

Searched for crucial parameters like learning rate, learning rate schedulers using [Optuna](https://optuna.org/) framework. 

## Instruction

Run [run_optuna.ipynb](run_optuna.ipynb).

## Optuna visualization

Optuna provides some useful insights on the searched parameters, you can visualize the study using the provided API; for instance, check the aforementioned notebook.

![importance](https://github.com/hrishikesh829370/Cassava_plant_disease_classifier/assets/131910887/7a574e67-9408-4925-a981-39b48c801091)
![history](https://github.com/hrishikesh829370/Cassava_plant_disease_classifier/assets/131910887/f85dd211-ff30-45e5-b9f2-2b5f0249c431)
![coordinates](https://github.com/hrishikesh829370/Cassava_plant_disease_classifier/assets/131910887/67bdefce-0ad0-4f1b-909f-a35cd3841407)

## Next Steps

This project is just the beginning. In the future, I plan to:

- Explore the use of Vision Transformer models like ViT and Deit for potential improvements.
- Investigate other loss functions, such as Taylor Loss and Bi-Tempered Loss, for possible enhancements.
- Continue refining the solution to further assist cassava farmers and promote food security in the world.
- deploying the model into a mobile application for the easy access by the farmers

## Get Involved

You can contribute to this project by providing ideas, code improvements, or collaborating on similar initiatives to support farmers in general. Together, we can make a significant impact on food security and crop health in Africa.

Let's work together to empower cassava farmers and make a difference!

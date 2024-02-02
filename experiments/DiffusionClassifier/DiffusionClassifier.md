# Diffusion Classifier

---

## Install

Please refer to:    [**BasicReadMe**](https://github.com/huanranchen/AdversarialAttack/blob/main/README.md)

### Model Checkpoints

**CIFAR10 unconditional diffusion model for DiffPure**:      
https://drive.google.com/file/d/1zfblaZd64Aye-FMpus85U5PLrpWKusO4/view            
Put it into ./resources/checkpoints/DiffPure/32x32_diffusion.pth

**CIFAR10 WideResNet-70-16-dropout~(discriminative classifier used in DiffPure)**:        
https://github.com/NVlabs/DiffPure, "Data and pre-trained models" part.       
Put it into ./resources/checkpoints/models/WideResNet_70_16_dropout.pt

**Conditional diffusion model for diffusion classifier**           
We will share our checkpoints soon. Now you can train it by yourself.      

**ImageNet unconditional diffusion model for DiffPure**:      
https://drive.google.com/file/d/1zfblaZd64Aye-FMpus85U5PLrpWKusO4/view               
Put it into ./resources/checkpoints/DiffPure/256x256_diffusion_uncond.pt


---


## Experiments

All experiments codes are in *'./experiments/DiffusionClassifier'*. 


**DiffAttack.py**:  Attack DiffPure.         

**DiffusionAsClassifier**: Test robustness of diffusion classifier under AutoAttack+BPDA/Lagrange/DirectDifferentiate 

**DiffusionMaximizer**: Likelihood maximizer. A new diffusion purification method we proposed. See Sec 3.4 in our paper for detail. Could be combined with discriminative classifier.

**DirectAttack**: Direct differentiate through likelihood maximization.

**ObfuscatedGradient**: Measure the cosine similarity between the gradient of diffusion classifier and DiffPure. See Sec 4.4 in our paper for detail.

**OptimalDiffusionClassifier**: See Sec 3.3 for detail.

**stadv**: Measure the robustness under STAdv attack.

**TrainDiffusion**: Train conditional diffusion models.








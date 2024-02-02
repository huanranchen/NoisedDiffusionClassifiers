# Your Diffusion Model is Secretly a Certifiably Robust Classifier

---

## Install
This repo is build upon the repo of [Diffusion Classifier](https://github.com/huanranchen/DiffusionClassifier).
Please refer to this repo for checkpoint downloading and environment installing.


---

## Usage

### Code Framework

> attacks: Some attack algorithms. Including VMI, VMI-CW, CW, SAM, etc.      
> data: loader of CIFAR, NIPS17, PACS    
> defenses: Some defenses algorithm    
> experiments: Example codes    
> models: Some pretrained models   
> optimizer: scheduler and optimizer   
> tester: some functions to test accuracy and attack success rate   
> utils: Utilities. Like draw landscape, get time, HRNet, etc.     


### Experiments
Some demos and key experiments:
- DCTK.py. Run diffusion classifier with our time complexity reduction technique.
- APNDC_certify. Certify the APNDC.
- EPNDC_certify. Certify the EPNDC.

For other experiments, please refer to
```text
./experiments/DiffusionClassifier/certify
```

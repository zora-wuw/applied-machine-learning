Applied Machine Learning
# Why use ML
1. Learning New Physics
2. Capture Expensive Simulation
3. Complicated Multi-Physics interaction
4. Gains from updating mdoels with new data

| Methodology                                 | Purpose                                                                                     | Scientific Benefit                                                               |
|----------------------------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Physics-Informed Neural Networks (PINNs)     | Embeds physical laws (e.g., thermodynamics, mass balance) directly into loss functions.      | Prevents the model from outputting physically impossible results.                 |
| Differentiable Simulation                    | Infuses compiler techniques and ML gradients directly into biological or physical simulations.| Accelerates simulation speeds up to several hundredfold compared to manual iterations. |
| Surrogate Modeling                          | Trains lightweight ML models to mirror slow, heavy numerical simulations.                    | Allows researchers to run millions of scenario trials in real-time.               |

# Problem definition & success measure
## Considerations
- Goal in the real world; how to measure it (by proxy) to stay close to the original goal?
- Data driven first. Start from the simplist algorithm, think the cost in context, if it is worth it to use complex models.
- Communicate the results in the problem context.
## Concepts
Supervised, Unsupervised, Reinforce Learning  
Supervised Learning  
Neural Network Overview in PyTorch  
    - Structure
    - Workflow
    - Training
    - Evaluate
    
# Feature engineering & data cleaning
## Considerations
- Dirty data
- Data to numbers
- Data bias from real world.

## Techniques 
- Cleaning - nulls, text tags
- Encoding - One-hot, ...
- Scaling 
- Calibration 
- Rotation, mirroring etc

## Demo
Exoplanet data EDA, Scale, Augmentation


# Strength and weakness of different models
## Concepts
MLP, CNN, RNN
Fully connected layer
Activation function
Loss function
Gradient 
Learning rate
Optimizor

## Considerations
Cost to train, maintan 
Model complexity - Underfit / overfit
Gradient explosion/diminishing - look at VGG-16 structure?

## Techniques
- Train, validate, test split 
- Others like K-Fold, Cross-Validation, Strategiefied, etc 
- Evaluations - accuracy, f2, ROC ...
- Dropout layer, Maxpool
- Batch normalisation, residual connection

## Demo
Exoplanet MLP and CNN: load, train, evaluate

## Practice
- pneumonia data augmentation with vanilla CNN
- pneumonia add layers in vanilla CNN
- finetune on ResNet-18

# Model selection best practices

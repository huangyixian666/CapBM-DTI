# CapBM-DTI
CapBM-DTI: A novel capsule network-based method integrating BERT and MPNN for prediction of drug-target interaction

This is the code for CapBM-DTI.


### Requirements

- Python 3.x
- numpy
- scikit-learn
- RDKit
- Tensorflow
- keras

### Run

To run this code:

python CapBM_DTI.py --dti data/benchmark.txt --protein-descripter bert  --drug-descripter MPNN --model-name bert_MPNN_capsule_celegans --batch-size 64 -e 1000 -dp data

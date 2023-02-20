<div align="center">
<h1><b>CandidatesGenerator</b></h1>
<p>This is the source code for a candidate generator for entity linking.</p>
<p>It provides a simple API for design, train, and evaluate candidate generation model. You can use the base models to easily develop your own model.</p>
</div>

<div align="center">
<img alt="PyPI - Python Version" src="https://img.shields.io/badge/Python-3.9-blue">
<img alt="PyPI - PyToch Version" src="https://img.shields.io/badge/PyTorch-1.11.0-blue">

</div>

<h2>Quick start</h2>
<p>Start command:
python core.py --modelname "MODEL_NAME" --indexname "INDEX_NAME" --datafolder "./../Data" --batchsize "BATCH_SIZE" --epochs "EPOCHS" --lr "LEARNING_RATE" --setsize "SIZE OF CANDIDATES SET"</p>

<p>Remarks:
    - If you want to use the bi-encoder architecture include the substring "blink" in the model's name
    - If no subfolder in the Models folder with the same name as the model is found a new model will be instantiated and trained.
    - The Dataset on which the training and the evaluation is ran is the AIDA dataset
</p>


<h2>Structure</h2>

    ├── ...
    ├── Code                    # Source Code 
    │   ├── Helpers             
    │   │   ├── Gen_Entities_Embeddings_Index.py        # Script needed to get the embeddings from the Index              
    │   ├── Core.py                                     # Start script
    │   └── Attention_Encoder.py                        # The model used to translate mention embeddings to entity embeddings
    │   │
    │   └── ...
    │
    └── Data                    # Where all the Datasets are stored or will be saved 
    │   ├── Datasets            # The original Dataset from which the training data will be extracted
    │   ├── Results             # Where all the files of the processed raw dataset will be saved
    │   
    └── Models                  # Where the Trained model will be save (Or where you can put the models to load) 
    │   
    └── ...

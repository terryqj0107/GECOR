# GECOR: An End-to-End Generative Ellipsis and Co-reference Resolution Model for Task-Oriented Dialogue
Source code and dataset for the paper "GECOR: An End-to-End Generative Ellipsis and Co-reference Resolution Model for Task-Oriented Dialogue" (Jun Quan, Deyi Xiong, Bonnie Webber and Changjian Hu.  EMNLP, 2019)

## Data
Since there are no publicly available labeled data for the resolution of ellipsis and co-reference in dialogue, we manually annotate such a new dataset based on the public dataset CamRest676 from the restaurant domain.
This dataset contains the following json files:
1. CamRest676_annotated.json: the new annotated dataset for generative ellipsis and co-reference resolution research for task-oriented dialogue.
2. CamRest676.json: the original woz dialogue dataset, which contains the conversion from users and wizards, as well as a set of coarse labels for each user turn.
3. CamRestDB.json: the Cambridge restaurant database file, containing restaurants in the Cambridge UK area and a set of attributes.
4. CamRestOTGY.json: specific all the values the three informable slots can take.

### Our annotation ———— CamRest676_annotated.json
#### Date statistics
The CamRest676 dataset contains 676 dialogues, with 2,744 user utterances. After annotation, 1,174 ellipsis versions and 1,209 co-reference versions are created from the 2,744 user utterances. 1,331 incomplete utterances are created that they are an either ellipsis or co-reference version. 1,413 of the 2,744 user utterances are complete and not amenable to change. No new versions are created from these 1,413 utterances.

#### Dataset Split for Experiments
We split the new dataset into a training set (accounting for 80%) and validation set (accounting for 20%) which can be used for the stand-alone ellipsis/coreference resolution task and the multi-task learning of both the ellipsis/co-reference resolution and end-to-end task-oriented dialogue.

#### Annotation Specification
Annotation cases for user utterances can be summarized into the following three conventions:
*  If a user utterance contains an ellipsis or anaphor, we manually resolve the ambiguity of ellipsis or anaphor and supplement the user utterance with a correct expression by checking the dialogue context. In doing so, we create a pragmatically complete version for the utterance. If the utterance only contains an ellipsis and the ellipsis can be replaced with an anaphor, we create a co-reference version for it. Similarly, if the utterance only contains an anaphor and the anaphor can be omitted, we create an ellipsis version for the utterance.
*  If the user utterance itself is pragmatically complete, without any ellipsis or anaphora, we create an anaphor and ellipsis version for it if such a creation is appropriate.
*  If the utterance itself is complete and it is not suitable to create an ellipsis or anaphor version, we just do nothing.
In CamRest676_annotated.json, the key 'transcript' represent the origin user utterance, and the keys 'transcript_complete', 'transcript_with_ellipsis', 'transcript_with_coreference' respectively represent the complete version, ellipsis version, coreference version of the user utterance after our annotation.  

#### Data License
We make the new annotated dataset under the following licenses:
*  Attribution 4.0 International (CC BY 4.0) license. 
(License URL: https://creativecommons.org/licenses/by/4.0/)


## Code
GECOR Model and Task-Oriented Dialogue with GECOR.

### Requirements
python 3.6 <br>
numpy==1.14.3 <br>
nltk==3.2.1 <br>
torch==0.3.1 <br>
```
pip install -r requirements.txt
```
Download pretrained Glove word vectors and place them in PROJECT_ROOT/data/glove. 


### Train
```
python model.py -mode train -model gecor-camrest
```
To train and test with different version of data (ellipsis/ co-reference/ mixed): <br>
please modify the code in config.py and reader.py

To switch between the GECOR1 mode and GECOR2 mode: <br>
please modify the code in GECOR_model.py

### Test
```
python model.py -mode test -model gecor-camrest
```


## References
If you use the source codes or datasets included here in your work, please cite the corresponding papers. The bibtex are listed below:
```
"GECOR: An End-to-End Generative Ellipsis and Co-reference Resolution Model for Task-Oriented Dialogue" 
(Jun Quan, Deyi Xiong, Bonnie Webber and Changjian Hu.  EMNLP, 2019)

We will soon upload the bibtex.

```

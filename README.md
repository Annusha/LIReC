# LIReC
[web-page](https://annusha.github.io/LIReC/)

Official implementation in python.  https://arxiv.org/pdf/2003.13158.pdf

If you use the code, please cite


```
@inproceedings{kukleva2020lirec,
  title={Learning Interactions and Relationships between Movie Characters},
  author={Kukleva, Anna and Tapaswi, Makarand and Laptev, Ivan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR'20)},
  year={2020}
}
```

To run the code first download the data: [link]() // will be here soon  
Note that it's about 80GB. Preserve inner structure.

1. Set all the paths:
```
LIReC/utils/arg_pars.py
  --project_root: path to the LIReC project
  --data_root: path to the folder with data
  --store_root: just in case if you want to try training to store models (optional)
```
2. Install environment
```
conda create --name lirec --file requirements.txt
conda actiavte lirec
```
3. To resume models from checkpoints:
```
modality check (model there is only for all three modalities)
  python resume/modalities.py
  
  
multi-task learning for interactions and relationships
  python resume/int_rels.py
  
  
interactions and movie characters detection
  python resume/int_ch.py
  
  
interactions, relationships and movie characters detection
  python resume/int_rel_ch.py
```
Each file has option `sanity_check`. If it sets to True, you can quickly check if nothing breaks with the data paths and models.   
If it sets to False, test will be made on the entire dataset.

4. Movie character detection can be evaluated with model trained on ground truth or weakly trained model. Set to corresponding value `tr_correct` in 'resume/int_ch.py' or 'resume/int_rel_ch.py'.

5. No specific code for training these models, sorry. But you can find trainig function, all the losses and other details in the code. If any questions, just drop me an email.
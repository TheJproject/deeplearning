# deeplearning 02456

### Problem task: use autoencoders on timeseries data from Green Mobility fleet to see if they fit in a nice latent space
### Structure
Z-acceleration time sequences are represented as pandas series.<br>
We write a `Dataloader` class which drops ones containing not enough data points, as well as creating pytorch `PackedSequences` out of them all. `PackedSequence` is a timeseries which is padded to a certain size for reconstruction purposes but lets `torch.RNN` layers know not to consider the padding as something to compute on.<br>
The encoder itself is written in `RNN_AE`,a wrapper class for an encoder and decoder subclass<br>
```
RNN_enc
#trains RRN layer using using packed sequences
#returns unpacked
# forward pass unpacks sequences qhen training RNN
```

```
RNN_dec
#calls get_packed method we defined 
#returns packed
```
Initializable like regular torch model (inherits from `nn.Module()`).<br>
Old class: `AE_conv`, uses $1D$ convolutional layers, which do not work with packed sequences. 

### TODO:
- s'assurer que tout marche bien pour tout le monde, installez bien parquet sur vos env python (avec le `%pip install connerie` sur jupitre). virez la layer `rnn2` (et changez le encoded dans la forward pass) pour pouvoir run plus vite et attendez 6 ou 7 prints de loss avant de dire que samarsh.<br>
- utiliser le vielle `AE_conv`et faire des random codesinge-changements d'un hyperparameter à la fois pour voir avec lesquels ca amrche mieux<br>
- si nécessaire: changer le dataloader  (utiliser juste le vieux de DL.py que vous avez sur vos ordis pour run sans les packedsequences)<br>
- réecrire la classe  `RNN_AE` (et aussi `AE_conv` tant qu'on y est) pour avoir des hyperparameters changeables au lieu des 5, 1, 4, etc arbitraires<br>
- plotter le loss même si il est dégueulasse<br>
- faire dessiner à l'ordinateur les reconstitutions du AE comme ca on peut se foutre de sa gueule parce qu'il est nul et gros et qu'il a un ptit zizi (plot)<br>
- save l'AE entrainé si possible une fois bien tuné. si vous faites des pickle mettez les en parquet je touche plus au moindre pickle wallah<br> 
- wrapper function pour TSNE (pas oublier de call `@torch.no_grad()` pour pas qu'il train le modèle alors qu'on lui fait manger des observations pour qu'il les chie dans le latent space)<br>
- TSNE avec nouveau AE. utiliser la focntion `get_latent` de la classe <br> 
- spectral analysis/clustering de scikit learn<br>
-d'autres visualizations si vs conaissez ¯\\\_(ツ)\_/¯ <br>
- la classification jsp comment on va faire mais ca devrait être possible et on mettra tous les graphiques cringes qui vont avec<br>
- j'espère que spectral analysis/clustering terraformera les fesses aux autoencodeurs parce qu'il est plus mieux


### Considérations pour le report:

- quelle Se2seq loss function à-t-on utilisé (ici norme L²), pq, pq pas une autre?

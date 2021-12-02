# deeplearning 02456

### Problem task: use autoencoders on timeseries data from Green Mobility fleet to see if they fit in a nice latent space

### TODO:
- utiliser le vielle `AE_conv`et bruteforce des changements d'un hyperparameter à la fois pour voir avec lesquels ca amrche mieux<br>
- réecrire la classe  `RNN_AE` (et aussi `AE_conv` tant qu'on y est) pour avoir des hyperparameters changeables au lieu des 5, 1, 4, etc arbitraires<br>
- plotter le loss même si il est dégueulasse<br>
- faire dessiner à l'ordinateur les reconstitutions du AE comme ca on peut se foutre de sa gueule parce qu'il est nul et gros et qu'il a un ptit zizi (plot)<br>
- save l'AE entrainé si possible une fois bien tuné. si vous faites des pickle mettez les en parquet je touche plus au moindre pickle wallah<br> 
- wrapper function pour TSNE (pas oublier de call `@torch.no_grad()` pour pas qu'il train le modèle alors qu'on lui fait manger des observations pour qu'il les chie dans le latent space)<br>
- TSNE avec nouveau AE<br> 
- spectral analysis/clustering de scikit learn

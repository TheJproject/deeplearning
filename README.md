# deeplearning 02456

### Problem task: use autoencoders on timeseries data from Green Mobility fleet to see if they fit in a latent space which could help in the binary classification problem: "good road/bad road"
### Structure
Z-acceleration time sequences are represented as pandas series.<br>
We do some preprocessing: drops ones containing not enough data points, as well as creating pytorch `PackedSequences` out of them all. `PackedSequence` is a timeseries which is padded to a certain size for reconstruction purposes but lets `torch.RNN` layers know not to consider the padding as something to compute on.<br>
We then load this data into an enhanced `Dataloader` object<br>
The encoder itself is written in `RNN_AE`,a wrapper class for an encoder and decoder subclass<br>
`newAE_conv.ipynb` is where you will find most reults: training and validation performance<br>
results of the visualization of the latent space are in the `visualizations.ipynb` file.
<br><br>
In general, when looking through old commit or old models we made, we tried to keep the conventions
```
*_enc (RNN_enc, enc, conv1d_enc, etc)
#trains RRN layer using using packed sequences
#returns unpacked
# forward pass unpacks sequences qhen training RNN
```

```
*_dec
#calls get_packed method we defined 
#returns packed
```
Initializable like regular torch model (inherits from `nn.Module()`).<br>
class: `AE_conv`, uses $1D$ convolutional layers, which do not work with packed sequences. 

### TODO:
ok :)
### Considérations pour le report:



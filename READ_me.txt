code files and outputs:

1) Training the autoencoder:

------There are two CAE models:
First model is in models/your_net.py
Second model is in models/newcae2.py


------To run the training:
First change the parameters of training in the config file.

--- To train 
the first model with normalized dataset is trained in train_1.py
the first model without normalization is trained in train_1_nonorm.py
The model returns two values decoded and encoded


the second model is trained in train_2.py

2) Testing and clustering 

the folder 'res' contains all the models trained
the final models are 

To obtain the clusters based on encodings of the first model run the test_mslp_withoutminmax.py 
it will visualize the clusters with no min max scale values 
 
To obtain clusters for the second model run test_era.py
There the dataset of choice should be uncommented and titles of output should be changed.

to link the clusters between each other run clusters_linking.py 
it takes txt files of clusters obtained by space and latent space clustering. 

Visualized clusters of the first model are in the folder "cluster_vis8"
where "latex" means it was edited for the paper

Visualized clusters of the second model for the ERA-20C dataset are in the folder "era20_clusters_withoutminmax"
Visualized clusters of the second model for the CMIP6 dataset are in the folder "CMIP_clusters"

3) MaskLayer results
first maskLayer should be true in config file and code sizes should be given 
the final model used for visualizing the MaskLayer results is in "AEcode/res/nonorm_masklayer_12_50ep/epoch_50.ckpt"
the directory should be put in the config file if no training is needed.
Run the mask_test.py, change the namings of the files to be saved if needed.

Than subtract the outputs of the saved files of mask_test.py run the file subtractLayers.py
it is not automated so for the subtraction, files should be chosen manually.
This will visualize every value of the latent space.

the results are in the folder "MaskLtxt"

4) Pre-processing of ERA-20C and CMIP6 data:

It was done using CDO command line. 
 For example processing of the future dataset:
1. cdo mulc,9.80665 zg500_clip.nc z500_clip.nc
2. cdo monmean z500_clip.nc monmean_z500.nc
3. cdo setday,1 monmean_z500.nc monmean_z500day1.nc
4. cdo -inttime,2015-01-01,12:00:00,1day monmean_z500day1.nc daily_z500.nc
5. cdo -monadd -gtc,100000000000000000 daily_z500.nc monmean_z500day1.nc final_z500daily.nc
choose only december of the last year 
6. cdo selyear,2100 z500_clip.nc z500_2100.nc
   cdo selmon,12 z500_2100.nc z500_210012.nc
7. Python copy last day of final_z500daily.nc to z500_210012.nc
8. delete last day in final_z500daily.nc
   cdo delete,timestep=31381 final_z500daily.nc final_z500dailynodec.nc
9. cdo mergetime final_z500dailynodec.nc z500_210012.nc final_z500dailyfull.nc 
10. cdo sub z500_clip.nc final_z500dailyfull.nc anomal_z500.nc
 then with cdo the data was spliited into three sets. 

5) EOF functions were obtained with CDO 

Results are in the folder "eof"
cdo select,startdate=1850-01-01,enddate=1972-12-31 mslp.nc mslp_train.nc  (1850 -1972 years - process the data)
cdo eof,10 mslp_test.nc eingenvals_mslp.nc eofs_mslp.nc

from here:
cdo -s eof,10 mslp_train.nc mslp_train_eigvals.nc mslp_train_eofs.nc

#get eof coefficients from test dataset:
cdo eofcoeff eofs_mslp.nc mslp_test.nc mslp_test_eofcoef 
will result in 10 coeffs
The reconstruction is then simply the sum of the coefficients times the corresponding eofs:
select first eof:
cdo seltimestep,1 mslp_train_eofs.nc mslptrain_eof1.nc
multiply coef with eof:
 create a copy to change it:
cdo copy mslp_test.nc mslptest_muleof1.nc
multiply in changenc in Python
repeat for each eof 
then add all 10 eofs multiplied with coefs done in changenc saved in mslptest_muleof1.nc




 
from cleanfid import fid

score = fid.compute_fid('./cifar_test/', '.cifar_train/', mode ="legacy_pytorch", ba
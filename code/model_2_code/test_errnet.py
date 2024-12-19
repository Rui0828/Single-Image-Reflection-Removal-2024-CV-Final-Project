from os.path import join, basename
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True
opt.display_id=0
opt.verbose = False

datadir = 'C:\\Users\\asd47\\OneDrive - rui0828\\master\\CV\\Final_Project\\Code\\paper_2\\ERRNet-master\\'

# Define evaluation/test dataset

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'test_data/testdata_reflection_synthetic_table2'))


eval_dataset_real = datasets.CEILTestDataset(join(datadir, 'test_data/Berkeley real dataset'))


eval_dataset_solidobject = datasets.CEILTestDataset(join(datadir, 'test_data/SIR2/SolidObjectDataset'))
eval_dataset_postcard = datasets.CEILTestDataset(join(datadir, 'test_data/SIR2/Postcard Dataset'))
eval_dataset_wild = datasets.CEILTestDataset(join(datadir, 'test_data/SIR2/Wildscene'))

eval_dataset_nature = datasets.CEILTestDataset(join(datadir, 'test_data/Nature'))

eval_dataset_nrd = datasets.CEILTestDataset(join(datadir, 'test_data/Natural Reflection Dataset(NRD)'))

# test_dataset_internet = datasets.RealDataset(join(datadir, 'internet'))
# test_dataset_unaligned300 = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned300/blended'))
# test_dataset_unaligned150 = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned150/blended'))
# test_dataset_unaligned_dynamic = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned_dynamic/blended'))
# test_dataset_sir2 = datasets.RealDataset(join(datadir, 'sir2_wogt/blended'))


eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_wild = datasets.DataLoader(
    eval_dataset_wild, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_nature = datasets.DataLoader(
    eval_dataset_nature, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_nrd = datasets.DataLoader(
    eval_dataset_nrd, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_internet = datasets.DataLoader(
#     test_dataset_internet, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_sir2 = datasets.DataLoader(
#     test_dataset_sir2, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned300 = datasets.DataLoader(
#     test_dataset_unaligned300, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned150 = datasets.DataLoader(
#     test_dataset_unaligned150, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned_dynamic = datasets.DataLoader(
#     test_dataset_unaligned_dynamic, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)


engine = Engine(opt)

"""Main Loop"""
if __name__ == '__main__':
    result_dir = './results'

    # evaluate on synthetic test data from CEILNet
    print('Testing CEILNet_table2')
    res1 = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2', savedir=join(result_dir, 'CEILNet_table2'))
    
    # print('Testing real')
    # res2 = engine.eval(eval_dataloader_real, dataset_name='real', savedir=join(result_dir, 'real'))

    print('Testing solidobject')
    res3 = engine.eval(eval_dataloader_solidobject, dataset_name='solidobject', savedir=join(result_dir, 'solidobject'))


    print('Testing postcard')
    res4 = engine.eval(eval_dataloader_postcard, dataset_name='postcard', savedir=join(result_dir, 'postcard'))

    print('Testing wild')
    res5 = engine.eval(eval_dataloader_wild, dataset_name='wild', savedir=join(result_dir, 'wild'))

    print('Testing nature')
    res6 = engine.eval(eval_dataloader_nature, dataset_name='nature', savedir=join(result_dir, 'nature'))

    # print('Testing NRD')
    # res7 = engine.eval(eval_dataloader_nrd, dataset_name='nrd', savedir=join(result_dir, 'nrd'))


    print("##############################################################################################################")
    print("Testing CEILNet_table2: ", res1)
    # print("Testing real: ", res2)
    print("Testing solidobject: ", res3)
    print("Testing postcard: ", res4)
    print("Testing wild: ", res5)
    print("Testing nature: ", res6)
    # print("Testing NRD: ", res7)
    print("##############################################################################################################")





# evaluate on four real-world benchmarks
# res = engine.eval(eval_dataloader_real, dataset_name='testdata_real')

# res = engine.eval(eval_dataloader_real, dataset_name='testdata_real', savedir=join(result_dir, 'real20'))
# res = engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=join(result_dir, 'postcard'))
# res = engine.eval(eval_dataloader_wild, dataset_name='testdata_sir2', savedir=join(result_dir, 'sir2_withgt'))
# res = engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=join(result_dir, 'solidobject'))

# test on our collected unaligned data or internet images
# res = engine.test(test_dataloader_internet, savedir=join(result_dir, 'internet'))
# res = engine.test(test_dataloader_unaligned300, savedir=join(result_dir, 'unaligned300'))
# res = engine.test(test_dataloader_unaligned150, savedir=join(result_dir, 'unaligned150'))
# res = engine.test(test_dataloader_unaligned_dynamic, savedir=join(result_dir, 'unaligned_dynamic'))
# res = engine.test(test_dataloader_sir2, savedir=join(result_dir, 'sir2_wogt'))

import os
import sys
import cv2
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import argparse


prob_layer = 'softmax_output'
arg_fc = 'fc'


class ModelParam:
    def __init__(self,model_name,model_path,model_index,conv_layer):
	self.model_name = model_name
	self.model_path = model_path
	self.model_index = model_index
	self.conv_layer = conv_layer


    def load_model(self, args):
	self.sym, self.arg_params, self.aux_params = mx.model.load_checkpoint(self.model_path,self.model_index)
	internals = self.sym.get_internals()
	print internals.list_outputs()[-20:-1]
	symbol = mx.sym.Group([internals[prob_layer], internals[self.conv_layer]])
	self.mod = mx.mod.Module(symbol, context=mx.gpu(args.gpu))
	self.mod.bind(data_shapes=[('data', (1, 3, 224, 224))], for_training=False)  # bangding
	self.mod.set_params(self.arg_params, self.aux_params)
	self.weight_fc = self.arg_params[arg_fc + '_weight'].asnumpy()



# MODEL_BASE_PATH = '/mnt/data/wanglichun/wlc/Output/'
# model_params = []
# model_params.append(ModelParam('densenet-169',os.path.join(MODEL_BASE_PATH,'output-fuliao/densenet-169-sample0.3-namihe-2018-3-20-v2rec-param13/densenet-169'),13,'DBstage4_concat32_output'))
# model_params.append(ModelParam('resnet-101',os.path.join(MODEL_BASE_PATH,'output-fuliao/resnet-101-sample0.3-namihe-2018-3-20-v2rec-param19/resnet-101'),19,'stage4_unit3_conv3_output'))


## fuliao
# MODEL_BASE_PATH = '/mnt/data/wanglichun/wlc/Output/'
# model_params = []
# model_params.append(ModelParam('densenet-169',os.path.join(MODEL_BASE_PATH,'output-fuliao/0611/densenet-169'),20,'DBstage4_concat32_output'))
# model_params.append(ModelParam('resnet-101',os.path.join(MODEL_BASE_PATH,'output-fuliao/0611/resnet-101'),17,'stage4_unit3_conv3_output'))
# model_params.append(ModelParam('resnext-50',os.path.join(MODEL_BASE_PATH,'output-fuliao/0611/resnext-50'),17,'stage4_unit3_conv3_output'))


### common
MODEL_BASE_PATH = '/mnt/data/wanglichun/wlc/Output/'
model_params = []
model_params.append(ModelParam('densenet-169',os.path.join(MODEL_BASE_PATH,'output-common/0518/densenet-169'),18,'DBstage4_concat32_output'))
model_params.append(ModelParam('resnet-101',os.path.join(MODEL_BASE_PATH,'output-common/0518/resnet-101'),17,'stage4_unit3_conv3_output'))
model_params.append(ModelParam('resnext-50',os.path.join(MODEL_BASE_PATH,'output-common/0518/resnext-50'),15,'stage4_unit3_conv3_output'))



### class instructions
synset_file = '/mnt/data/wanglichun/project/porno-classification/model/synset.txt'
synset = [l.strip() for l in open(synset_file).readlines()]





def get_cam(conv_feat_map, weight_fc):
    assert len(weight_fc.shape) == 2
    if len(conv_feat_map.shape) == 3:
        C, H, W = conv_feat_map.shape
        assert weight_fc.shape[1] == C
        detection_map = weight_fc.dot(conv_feat_map.reshape(C, H*W))
        detection_map = detection_map.reshape(-1, H,W)
    elif len(conv_feat_map.shape) == 4:
        N, C, H, W = conv_feat_map.shape
        assert weight_fc.shape[1] == C
        M = weight_fc.shape[0]
        detection_map = np.zeros((N, M, H, W))
        for i in xrange(N):
            tmp_detection_map = weight_fc.dot(conv_feat_map[i].reshape(C, H*W))
            detection_map[i, :, :, :] = tmp_detection_map.reshape(-1, H, W)
    return detection_map


def data_transpose(data, augments):
    for aug in augments:
	data = aug(data)
    return data

def image_aug(img):

    cla_cast_aug = mx.image.CastAug()
   # cla_resize_aug = mx.image.ForceResizeAug(size=[224, 224])
    color_norm_aug = mx.image.ColorNormalizeAug(mx.nd.array([123, 117, 104]), mx.nd.array([1, 1, 1]))
    cla_augmenters = [cla_cast_aug, color_norm_aug]

    img = data_transpose(img,cla_augmenters)
    img = mx.nd.transpose(img,axes=(2,0,1))
    img = mx.nd.expand_dims(img,axis=0)
    return img


def main(args):

    ##### load models
    for item in model_params:
	item.load_model(args)


    if not os.path.exists(args.test_fold_path):
	print('test_fold_path is not exists, please check it ')
	return
    im_names = os.listdir(args.test_fold_path)  ### read image names

    for im_name in im_names:
	im_file = os.path.join(args.test_fold_path, im_name)
	if os.path.isdir(im_file):
	    continue
	print 'processing : ' + im_file



	im = cv2.imread(im_file)
	if im is None:
	    print('image read error:' + im_file)
	    continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	rgb = cv2.resize(rgb,(args.width,args.height))
	blob = image_aug(mx.nd.array(rgb))

	data_batch = mx.io.DataBatch([mx.nd.array(blob)])

	plt.figure(figsize=(18,args.model_total_num * 6))
	plt.subplot(args.model_total_num, 1 + args.top_n, 1)


	for model_num in range(args.model_total_num):


	    outputs = model_params[model_num].mod.forward(data_batch)

	    score = model_params[model_num].mod.get_outputs()[0].asnumpy()[0]  ## return is list of NDArray  so  we shuold use mod.get_outputs()[0]
		# print score
	    conv_fm = model_params[model_num].mod.get_outputs()[1].asnumpy()[0]
		# print conv_fm


	    score_sort = -np.sort(-score)[:args.top_n]
	    inds_sort = np.argsort(-score)[:args.top_n]

		# In[98]:
	    plt.subplot(args.model_total_num, 1 + args.top_n, 1 + ((args.top_n + 1) * model_num))
	    plt.imshow(rgb)
	    plt.title('model:{}'.format( model_params[model_num].model_name) )


	    cam = get_cam(conv_fm, model_params[model_num].weight_fc[inds_sort, :])
	    for k in xrange(args.top_n):
		detection_map = np.squeeze(cam.astype(np.float32)[k, :, :])
		heat_map = cv2.resize(detection_map, (args.width, args.height))
		max_response = detection_map.mean()
		heat_map /= heat_map.max()

		im_show = rgb.astype(np.float32) / 255 * 0.3 + plt.cm.jet(heat_map / heat_map.max())[:, :, :3] * 0.7
		plt.subplot(args.model_total_num , 1 + args.top_n, k + 2 + ((args.top_n + 1) * model_num))
		plt.imshow(im_show)
		plt.title('class:{},prob:{:.3}'.format(synset[inds_sort[k]], score_sort[k]))
		print 'Top %d: %s(%.6f), max_response=%.4f' % (k + 1, synset[inds_sort[k]], score_sort[k], max_response)


        save_path = os.path.join(args.test_fold_path, 'result' + args.add_tag)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path , 'hot_' + im_name) )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-total-num', type=int, default = 3)
    parser.add_argument('--test-fold-path',type=str,default='/mnt/data/wanglichun/wlc/Data/Hot-images/20180625-test')
    parser.add_argument('--top-n',type=int,default=3,help='the top th class ,that you want to show')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--add-tag', type=str, default='resnet')
    args = parser.parse_args()
    main(args)

import os
import sys
import time
import torch
import random
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision import transforms 
from torch.utils.data.sampler import SequentialSampler
import dct 
import freq_res


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
	
	return filename.lower().endswith(extensions)


def is_image_file(filename):
	
	return has_file_allowed_extension(filename, IMG_EXTENSIONS)

	
class BinaryNet(nn.Module):
	def __init__(self):
		super(BinaryNet, self).__init__()
		self.features = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])
		self.fc = nn.Sequential(
			nn.Linear(2048, 512),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, 2)
			)
	def forward(self, x):
		out = self.features(x)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		out = F.softmax(out, dim=1)
		conf_fake = out[:,1]
		conf_fake[conf_fake>0.1] = (5*conf_fake[conf_fake>0.1]+4)/9
		conf_fake[conf_fake<0.1] = 5*conf_fake[conf_fake<0.1]
		conf_real = 1-conf_fake
		out[:,0] = conf_real
		out[:,1] = conf_fake
		conf, predicted = torch.max(out,1)
		return conf, predicted


#def evaluate(data_path, model_load_path="", num_workers=1,  batch_size=1, use_cuda=True, output_filepath="results.txt"):
def evaluateEVAL2(filename, model_load_path="", num_workers=1,  batch_size=1, use_cuda=True, output_filepath="results.txt"):

	st = time.time()

	#if not os.path.exists(data_path):
	#	raise Exception("Data directory {} not found".format(data_path))

	if model_load_path and not os.path.exists(model_load_path):
		raise Exception("Model path {} not found".format(model_load_path))

	if not model_load_path:
		model_load_path = "models/model_100.pth"
		if not os.path.exists(model_load_path):
			raise Exception("Model path {} not found".format(model_load_path))

	resize_dims = 256

	transform_img = transforms.Compose([
		transforms.Resize(resize_dims),
		transforms.CenterCrop(resize_dims),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5],
							 std=[0.5, 0.5, 0.5] )
		])

	device = torch.device("cpu") if not use_cuda else torch.device("cuda:0")
	checkpoint = torch.load(model_load_path, map_location=device)
	model = nn.DataParallel(BinaryNet()).to(device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	#output_file = open(output_filepath,"w+")

	#filenames = [os.path.join(data_path,f) for f in os.listdir(data_path)] if not is_image_file(data_path) else [data_path]
	img_tensor = None
	#for i,filename in enumerate(filenames):
	if 1:
		#sys.stdout.write('\rLoading image {}/{}'.format(i+1,len(filenames)))
		if is_image_file(filename):
			img_pil = default_loader(filename)
			if img_tensor is None:
				img_tensor = transform_img(img_pil).unsqueeze(0)
				img_names = [filename]
			else:
				img_tensor = torch.cat((img_tensor,transform_img(img_pil).unsqueeze(0)),dim=0)
				img_names.append(filename)

		#if len(img_names)==batch_size or i==len(filenames)-1:
		if 1:
			with torch.no_grad():
				conf, predicted = model(img_tensor)

			for b in range(len(img_names)):
				if predicted[b]==0:
					print_str = "Image {} is real with confidence {:.4f}\n".format(img_names[b], conf[b].item())
				else:
					print_str = "Image {} is fake with confidence {:.4f}\n".format(img_names[b], conf[b].item())
				#output_file.write(print_str)
				print(print_str)

			img_tensor = None
			img_names = []

	#output_file.close()
	print("Predicted " + str(float(predicted[0])))
	print('Total time taken for evaluation: {0:.2f} s'.format(time.time()-st))
	return float(predicted[0]), conf[0].item()


#################################################################below added for Eval 324
class CoefficientShuffler(torch.nn.Module):
    def __init__(self, channels, direction='channels'):
        super(CoefficientShuffler, self).__init__()
        self._channels = channels
        self._direction = direction
    def forward(self, x, pad=None):
        if self._direction == 'channels':
            return self.channels(x)
        elif self._direction == 'blocks':
            return self.blocks(x, pad)
    def channels(self, x):
        blocks = torch.nn.functional.unfold(x, kernel_size=8, stride=8)
        blocks = blocks.transpose(1, 2).contiguous().view(-1, x.shape[2] // 8, x.shape[3] // 8, self._channels, 64)
        blocks = blocks.transpose(2, 3).transpose(1, 2)
        blocks = blocks.transpose(3, 4).transpose(2, 3).transpose(1, 2)
        blocks = blocks.contiguous().view(-1, 64 * self._channels, x.shape[2] // 8, x.shape[3] // 8)
        return blocks
    def blocks(self, x, pad):
        # This is just the inverse procedure from channels
        blocks = x.view(-1, 64, self._channels, x.shape[2], x.shape[3])
        blocks = blocks.transpose(1, 2).transpose(2, 3).transpose(3, 4)
        blocks = blocks.transpose(1, 2).transpose(2, 3)
        blocks = blocks.contiguous().view(-1, x.shape[2] * x.shape[3], self._channels * 64)
        blocks = blocks.transpose(1, 2)
        blocks = torch.nn.functional.fold(blocks, kernel_size=8, stride=8, output_size=(x.shape[2] * 8, x.shape[3] * 8))
        if pad is not None:
            diffY = pad.shape[2] - blocks.shape[2]
            diffX = pad.shape[3] - blocks.shape[3]
            blocks = torch.nn.functional.pad(blocks, pad=(diffX // 2, diffX - diffX // 2,
                                                          diffY // 2, diffY - diffY // 2))
        return blocks

class CoeffShuffle(object):
    def __init__(self):
        self.coeff_convert = CoefficientShuffler(3,direction='channels')
        schema = np.arange(64).reshape(8,8)
        schema = dct.zigzag(schema)
        self.schema = torch.tensor(schema).long()
    def __call__(self, tensor):
        device = tensor.device
        tensor = dct.batch_dct(tensor.unsqueeze(0), device=device)
        tensor = self.coeff_convert(tensor)
        n,c,h,w = tensor.shape
        tensor = tensor.view(n,64,c//64,h,w)
        tensor = tensor[:,self.schema.to(tensor.device),:,:,:]
        tensor = tensor.view(n,c,h,w)
        return tensor.squeeze()

class MulticlassNet(nn.Module):
    def __init__(self, num_classes):
        super(MulticlassNet, self).__init__()
        self.features_spatial = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.features_frequency = nn.Sequential(*list(freq_res.resnet50().children())[:-1])
        self.concat = nn.Linear(4096, num_classes)

    def forward(self, x):
        spatial = self.features_spatial(x['spatial'])
        spatial = spatial.reshape(spatial.size(0), -1)

        frequency = self.features_frequency(x['frequency'])
        frequency = frequency.reshape(frequency.size(0), -1)
        concat = torch.cat((spatial, frequency),dim=1)
        out = self.concat(concat)
        return out



###########################################################################################
def evaluate324(filename, model_load_path="", num_workers=1, batch_size=1, use_cuda=True, output_filepath="results.txt"):

    st = time.time()

    #if not os.path.exists(data_path):
        #raise Exception("Data directory {} not found".format(data_path))

    if model_load_path and not os.path.exists(model_load_path):
        raise Exception("Model path {} not found".format(model_load_path))

    if not model_load_path:
        model_load_path = "models/checkpoint.pth"
        if not os.path.exists(model_load_path):
            raise Exception("Model path {} not found".format(model_load_path))

    resize_dims = 256
    transform_img = [transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.CenterCrop(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        ]), transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.CenterCrop(resize_dims),
        transforms.ToTensor(),
        CoeffShuffle(),
        transforms.Normalize(mean=[0.5]*192,std=[0.5]*192)
        ])]

    test_classes = sorted(["guided-diffusion", "Latent-diffusion", "LSGM", "StyleGAN2", "StyleGAN3", "Taming-transformers"])
    #ORGINAL
    test_classes = sorted(["guided_diffusion", "latent_diffusion", "LSGM", "stylegan2", "stylegan3", "taming"])

    device = torch.device("cpu") if not use_cuda else torch.device("cuda:0")
    checkpoint = torch.load(model_load_path, map_location=device)
    num_classes = len(test_classes)
    model = nn.DataParallel(MulticlassNet(num_classes=num_classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #output_file = open(output_filepath,"w+")

    #filenames = [os.path.join(data_path,f) for f in os.listdir(data_path)] if not is_image_file(data_path) else [data_path]
    img_tensor = {}
    #for i,filename in enumerate(filenames):
    if 1:
        #sys.stdout.write('\rLoading image {}/{}'.format(i+1,len(filenames)))
        if is_image_file(filename):
            img_pil = default_loader(filename)
            if len(img_tensor)==0:
                img_tensor['spatial'] = transform_img[0](img_pil).unsqueeze(0)
                img_tensor['frequency'] = transform_img[1](img_pil).unsqueeze(0)
                img_names = [filename]
            else:
                img_tensor['spatial'] = torch.cat((img_tensor['spatial'],transform_img[0](img_pil).unsqueeze(0)),dim=0)
                img_tensor['frequency'] = torch.cat((img_tensor['frequency'],transform_img[1](img_pil).unsqueeze(0)),dim=0)
                img_names.append(filename)

        #if len(img_names)==batch_size or i==len(filenames)-1:
        if 1:

            with torch.no_grad():
                outputs = F.softmax(model(img_tensor),dim=1)
                conf, predicted = torch.max(outputs.data, 1)

            for b in range(1) : # range(len(img_names)):
                print_str = "Image {} belongs to class {} with confidence {:.4f}\n"\
                            .format(img_names[b], test_classes[predicted[b]], conf[b].item())
                #output_file.write(print_str)
                print(print_str)

            img_tensor = {}
            img_names = []

    #output_file.close()
    #print('YY '); print(int(predicted[0])); 
    #print(test_classes)

    CLASS_ID=int(predicted[0]);  print(test_classes[CLASS_ID])
    print("Results "+ str( test_classes[predicted[0]])+'  ' + str( conf[0].item()))
    LABEL=str( test_classes[predicted[0]])
    if LABEL=="latent_diffusion":
       LABEL="Latent-diffusion"
    elif LABEL=="stylegan2":
       LABEL="StyleGAN2"
    elif LABEL=="stylegan3":
       LABEL="StyleGAN3"
    elif LABEL=="taming":
       LABEL="Taming-transformers"
    elif LABEL=="guided_diffusion":
       LABEL="Guided-diffusion"

    print("Results "+ LABEL+'  ' + str( conf[0].item()))
    #print('Total time taken for evaluation: {0:.2f} s'.format(time.time()-st))
    #return test_classes[predicted[0]], conf[0].item()
    return LABEL ,  conf[0].item()

if __name__ == "__main__":

	#data_path = "./data"
	#evaluate(data_path, model_load_path="", num_workers=4, batch_size=2, use_cuda=False, output_filepath = "data_binary.txt")
        ############## This evaluates one real and one generated image samples, returns 0.0 if real and 1.0 if generated
        evaluateEVAL2("./data/121__face1.png", model_load_path="", num_workers=1, batch_size=1, use_cuda=True, output_filepath = "data_binary.txt")
        # evaluateEVAL2("./data/generated.photos_v3_0013830.jpg", model_load_path="", num_workers=1, batch_size=1, use_cuda=True, output_filepath = "data_binary.txt")
        ############## This scores a whole folder of images into one of 6 categories of generators ["guided-diffusion", "Latent-diffusion", "LSGM", "StyleGAN2", "StyleGAN3", "Taming-transformers"] 
        # FLDR='/mnt/Eval3.2.4/Image/TamingTransformers_EXPORT/'; TAG="Taming-transformers"
        # list_dir = os.listdir(FLDR)
        # slist=sorted(list_dir)
        # print(len(slist)); assets=0; good=0; correct=0; count=0
        # for i in range(len(slist)):
        #     name=slist[i]; count+=1 
        #     L,score=evaluate324(FLDR+name, model_load_path="", num_workers=1, batch_size=1, use_cuda=True, output_filepath = "data_binary.txt")
        #     print("Image classified as "+L+" with score confidence="+str(score)); exit()
            #if L==TAG:
               #correct+=1

        #print("CORRECT " + str(correct)+'/'+str(count))

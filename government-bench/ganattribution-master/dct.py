import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision
from torchvision import transforms

import math
import numpy as np
import cv2
from PIL import Image 


#Fold and unfold help in applying a function in a sliding window fashion.
def blockify(im, size):
	bs = im.shape[0]
	ch = im.shape[1]
	h = im.shape[2]
	w = im.shape[3]

	im = im.view(bs * ch, 1, h, w)
	im = torch.nn.functional.unfold(im, kernel_size=size, stride=size)
	im = im.transpose(1, 2)
	im = im.view(bs * ch, -1, size, size)

	return im


def deblockify(blocks, ch, size):
	bs = blocks.shape[0] // ch
	block_size = blocks.shape[2]

	blocks = blocks.reshape(bs * ch, -1, block_size**2)
	blocks = blocks.transpose(1, 2)
	blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=block_size, stride=block_size)
	blocks = blocks.reshape(bs, ch, size[0], size[1])

	return blocks


def normalize(N):
	n = torch.ones((N, 1))
	n[0, 0] = 1 / math.sqrt(2)
	return (n @ n.t())


def harmonics(N):
	spatial = torch.arange(float(N)).reshape((N, 1))
	spectral = torch.arange(float(N)).reshape((1, N))

	spatial = 2 * spatial + 1
	spectral = (spectral * math.pi) / (2 * N)

	return torch.cos(spatial @ spectral)


def block_dct(im, device=None):
	N = im.shape[3]

	n = normalize(N)
	h = harmonics(N)

	if device is not None:
		n = n.to(device)
		h = h.to(device)
	coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ im @ h)

	return coeff


def block_idct(coeff, device=None):
	N = coeff.shape[3]

	n = normalize(N)
	h = harmonics(N)

	if device is not None:
		n = n.to(device)
		h = h.to(device)
	im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
	return im

def convert_to_zigzag(dct_blocks,device=None,n_freq=None,block_freq=None):
	"""
	n_freq is number of freq components to include in the compression.
	Rest are set to zero. 
	The order of the freq in each block is obtained by zigzag function.
	n_freq is a tuple which says inclusively which all freqs to include. 
		range: 0 - 63 (inclusive)
	"""
	#changed to RERUN2
	# if n_freq is not None:
	# 	assert n_freq[0] >= 0 and n_freq[-1] <= 63

	if n_freq is None:
		n_freq = (0,63) #include all

	ch = dct_blocks.shape[0]
	N = dct_blocks.shape[1] #Number of 8X8 blocks in the image.

	size = (dct_blocks.shape[2],dct_blocks.shape[3])

	schema = np.arange(size[0]*size[1])
	schema = schema.reshape(size)
	schema = zigzag(schema)
	schema = torch.tensor(schema).long()
	#breakpoint()
	left_out_indices = schema[:n_freq[0]].tolist() + schema[n_freq[1]+1:].tolist()

	if block_freq is not None:
		if type(block_freq) ==int:
			left_out_indices.append(block_freq)
		elif type(block_freq) == list:
			left_out_indices += block_freq
		#print(left_out_indices)	

	schema_mask = torch.ones(size).flatten().to(device)
	schema_mask[left_out_indices] = 0 
	schema_mask = schema_mask.reshape(size,size)
	#Create an 8X8 mask with selected indices and multiply it with the blocks.

	#How to reorder without loop.
	dct_blocks = dct_blocks * schema_mask
	

	#d_block = dct_blocks.clone()	

	# for i in range(N):
	# 	block = dct_blocks[:,i,:,:]
	# 	#channelwise.
	# 	for c in range(ch):
	# 		b = block[c].flatten()
	# 
	# 		#b = b[schema]
	# 		b[left_out_indices] = 0
	# 		b = b.reshape(size,size)
	# 		#break
	# 		dct_blocks[c,i,:,:] = b
		#break


	return dct_blocks

def convert_inverse_zigzag(im_blocks,device=None):
	ch = im_blocks.shape[0]
	N = im_blocks.shape[1] #Number of 8X8 blocks in the image.

	size = (im_blocks.shape[2],im_blocks.shape[3])

	#pdb.set_trace()

	for i in range(N):
		block = im_blocks[:,i,:,:]
		#channelwise.
		for c in range(ch):
			b = block[c].flatten()
			#pdb.set_trace()
			b = inverse_zigzag(b,size[0],size[1],device)
			#b = torch.tensor(b).to(device)

			im_blocks[c,i,:,:] = b
	return im_blocks	


def batch_dct(im, device=None,n_freq=None,block_freq=None):

	# if device is not None:
	# 	im = im.to(device)

	"""
		n_freq is inclusive of both frequencies to include.

	"""

	ch = im.shape[1]
	size = (im.shape[2], im.shape[3])

	im_blocks = blockify(im, 8)
	dct_blocks = block_dct(im_blocks, device=device)

	#Select relevant top frequencies from zigzag orders.
	#We dont rearrage elements in zigzag. We just select the top freq based on zigzag order and
	#then do IDCT. Earlier we were doing zigzag reorder,selection followed by inverse!!

	dct_blocks = convert_to_zigzag(dct_blocks,device=device,n_freq=n_freq,block_freq=block_freq)

	dct = deblockify(dct_blocks, ch, size)

	return dct


def batch_idct(dct, device=None):
	ch = dct.shape[1]
	size = (dct.shape[2], dct.shape[3])

	dct_blocks = blockify(dct, 8)

	#dct_blocks = convert_inverse_zigzag(dct_blocks,device=device)
	#Remove zigzag conversion.

	im_blocks = block_idct(dct_blocks, device=device)

   

	im = deblockify(im_blocks, ch, size)

	return im

def per_freq_dct(input_tensor):
	"""
		Obtain all dct per frequency. Works fine for multibatch.
	"""
	device = input_tensor.device
	N,ch,H,W = input_tensor.shape 

	dct_img = torch.stack( [ batch_dct(input_tensor,n_freq=(i,i),device=device) for i in range(64) ] )
	dct_img = dct_img.to(device)

	return dct_img

def per_freq_idct(per_freq_dct):
	"""
		Invert the per freq dct and convert back to normal image (hopefully) :)
	"""
	device = per_freq_dct.device
	
	_,N,ch,H,W = per_freq_dct.shape 
	img_size = (H,W)
	# blocks_per_freq = torch.stack([ blockify(per_freq_dct[x],8) for x in range(64)])
	
	# #Each of these 8x8 blocks have their particular freq values, with others as zeros. 
	# #Need to collapse all the 8x8 blocks into one with all values "Filled in"
	# blocks_per_freq[blocks_per_freq==0]= 1
	# block = torch.prod(blocks_per_freq,0)
	# #block_inverse dct
	# img_blocks = block_idct(block,device=device)
	# #convert back to image
	# im = deblockify(img_blocks,ch,img_size)


	#Okayish method.
	#per_freq_dct[per_freq_dct==0] = 1
	#per_freq_dct = torch.prod(per_freq_dct,0)
	#im = batch_idct(per_freq_dct,device=device)


	#More efficient method.
	#breakpoint()
	mask = ~per_freq_dct.clone().bool()
	per_freq_dct = torch.where(per_freq_dct!=0,per_freq_dct,mask.float())
	per_freq_dct = torch.prod(per_freq_dct,0)
	im = batch_idct(per_freq_dct,device=device)

	return im




def to_ycbcr(x, device=None):
	ycbcr_from_rgb = torch.Tensor([
		0.29900, 0.58700, 0.11400,
		-0.168735892, -0.331264108, 0.50000,
		0.50000, -0.418687589, -0.081312411
	]).view(3, 3).transpose(0, 1).to(device)

	b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1).to(device)

	x = torch.einsum('cv,bcxy->bvxy', [ycbcr_from_rgb, x])
	x += b

	return x.contiguous()


def to_rgb(x, device=None):
	rgb_from_ycbcr = torch.Tensor([
		1, 0, 1.40200,
		1, -0.344136286, -0.714136286,
		1, 1.77200, 0
	]).view(3, 3).transpose(0, 1).to(device)

	b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1).to(device)

	x -= b
	x = torch.einsum('cv,bcxy->bvxy', [rgb_from_ycbcr, x])

	return x.contiguous()


def prepare_dct(dct, stats, device=None, type=None):
	ch = []

	for i in range(dct.shape[1]):
		dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

		t = ['y', 'cb', 'cr'][i] if type is None else type
		dct_blocks = stats.forward(dct_blocks, device=device, type=t)

		ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

	return torch.cat(ch, dim=1)


def unprepare_dct(dct, stats, device=None, type=None):
	ch = []

	for i in range(dct.shape[1]):
		dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

		t = ['y', 'cb', 'cr'][i] if type is None else type
		dct_blocks = stats.backward(dct_blocks, device=device, type=t)

		ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

	return torch.cat(ch, dim=1)


def batch_to_images(dct, stats, device=None, scale_freq=True, crop=None, type=None):
	if scale_freq:
		dct = unprepare_dct(dct, stats, device=device, type=type)

	spatial = batch_idct(dct, device=device) + 128

	if spatial.shape[1] == 3:
		spatial = to_rgb(spatial, device)

	spatial = spatial.clamp(0, 255)
	spatial = spatial / 255

	if crop is not None:
		while len(crop.shape) > 1:
			crop = crop[0]

		cropY = crop[-2]
		cropX = crop[-1]

		spatial = spatial[:, :, :cropY, :cropX]

	return spatial


def images_to_batch(spatial, stats, device=None, type=None):
	spatial *= 255

	if spatial.shape[1] == 3:
		spatial = to_ycbcr(spatial, device)

	spatial -= 128

	frequency = batch_dct(spatial, device=device)
	return prepare_dct(frequency, stats, device=device, type=type)
  


import numpy as np

def zigzag(input):
	#initializing the variables
	#----------------------------------
	h = 0
	v = 0
	vmin = 0
	hmin = 0
	vmax = input.shape[0]
	hmax = input.shape[1]
	#print(vmax ,hmax )
	i = 0
	output = np.zeros(( vmax * hmax))
	#----------------------------------

	while ((v < vmax) and (h < hmax)):
		if ((h + v) % 2) == 0:                 # going up
			if (v == vmin):
				output[i] = input[v, h]        # if we got to the first line
				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[i] = input[v, h] 
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[i] = input[v, h] 
				v = v - 1
				h = h + 1
				i = i + 1        
		else:                                    # going down
			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[i] = input[v, h] 
				h = h + 1
				i = i + 1
			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[i] = input[v, h] 

				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1

				i = i + 1

			elif ((v < vmax -1) and (h > hmin)):     # all other cases
				#print(6)
				output[i] = input[v, h] 
				v = v + 1
				h = h - 1
				i = i + 1

		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)        	
			output[i] = input[v, h] 
			break

	return output




# Inverse zigzag scan of a matrix
# Arguments are: a 1-by-m*n array, 
# where m & n are vertical & horizontal sizes of an output matrix.
# Function returns a two-dimensional matrix of defined sizes,
# consisting of input array items gathered by a zigzag method.
#
# Matlab Code:
# Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
# June 2007
# alex.nickel@gmail.com


def inverse_zigzag(input, vmax, hmax,device=None):

	# initializing the variables
	#----------------------------------
	h = 0
	v = 0
	vmin = 0
	hmin = 0
	#output = np.zeros((vmax, hmax))
	output = torch.zeros((vmax, hmax)).to(device)
	i = 0
	#----------------------------------
	while ((v < vmax) and (h < hmax)): 
		#print ('v:',v,', h:',h,', i:',i)   	
		if ((h + v) % 2) == 0:                 # going up            
			if (v == vmin):
				#print(1)				
				output[v, h] = input[i]        # if we got to the first line
				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[v, h] = input[i] 
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[v, h] = input[i] 
				v = v - 1
				h = h + 1
				i = i + 1        
		else:                                    # going down
			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[v, h] = input[i] 
				h = h + 1
				i = i + 1
			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[v, h] = input[i] 
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1
								
			elif((v < vmax -1) and (h > hmin)):     # all other cases
				output[v, h] = input[i] 
				v = v + 1
				h = h - 1
				i = i + 1

		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)        	
			output[v, h] = input[i] 
			break

	return output


def save_tensor(a,filename):
	a = a.cpu().detach().numpy()
	a = np.squeeze(a)
	try:
		a = np.transpose(a,(1,2,0))
	except:
		pass

	cv2.imwrite(filename,a*255.0)


if __name__ == '__main__':


	img = cv2.imread('../playground/img_49.png')
	img = np.transpose(img,(2,0,1))


	#img = transforms.ToTensor()(img)

	#img = np.array([[255,255,227,204,204,203,192,217],[215,189,167,166,160,135,167,244],[169,115,99,99,99,82,127,220],[146,90,86,88,84,63,195,189],[255,255,231,239,240,182,251,232],[255,255,21,245,226,169,229,247],[255,255,222,251,174,209,174,163],[255,255,221,184,205,248,249,220]])
	#img = np.ones((16,16))

	img = torch.tensor(img)
	#img.unsqueeze_(0)
	img.unsqueeze_(0)
	img = img.float()
	

	# dct = batch_dct(img,n_freq=(0,63))

	# dct_image = np.uint8(dct.numpy()[0][0]) #Same as before zigzag.

	# idct = batch_idct(dct)

	# #Very important
	# idct_image = np.uint8(idct.numpy())

	# idct_image = np.squeeze(idct_image)
	# idct_image = np.transpose(idct_image,(1,2,0))
	# cv2.imwrite('temp.png',idct_image)

	#Fucking works!!

	img = img/255.0

	mean = torch.tensor([0.5,0.5,0.5]).reshape(1,3,1,1)
	std = torch.tensor([0.5,0.5,0.5]).reshape(1,3,1,1)

	inv_mean = torch.tensor([-1,-1,-1]).reshape(1,3,1,1)
	inv_std = torch.tensor([2,2,2]).reshape(1,3,1,1)

	y_img = to_ycbcr(img)
	y_img_norm = y_img.sub_(mean).div_(std)
	y_img_dct = batch_dct(y_img_norm,device='cpu')


	#Reverse.

	rgb_img = to_rgb(y_img)



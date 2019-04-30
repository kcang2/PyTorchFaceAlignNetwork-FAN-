import os.path as op
import torch
from torchvision import transforms
from torchvision.transforms import functional as txf
import random
import cv2
import numpy as np
from skimage import io
import skimage.util as sku
from torch.utils.data import Dataset, DataLoader
import heatmap as hm

## TODO: Might want to use torchvision's transforms for all augmentations ##

class FaceDataset(Dataset):
	def __init__(self, root, transform=None, is_train=False):
		self.is_train = is_train
		self.transform = transform
		self.annot_root = op.join(root, "WFLW_annotations", "list_98pt_rect_attr_train_test")
		self.img_root = op.join(root, "WFLW_images")

		if is_train:
			filename = op.join(self.annot_root, "list_98pt_rect_attr_train.txt")
		else:
			filename = op.join(self.annot_root, "list_98pt_rect_attr_test.txt")
		
		self.im_list = []
		self.lm_mat = []
		self.bbox_mat = []	
		f = open(filename, "r").readlines()
		self.length = len(f)
		face = [54,76,82,96,97]	 # List of relevant landmarks
		for l in f:
			l = l.split()
			self.lm_mat.append(l[:196])
			self.bbox_mat.append(l[196:200])
			self.im_list.append(l[-1])
		self.lm_mat = np.array(self.lm_mat).reshape((-1, 98, 2)).astype(np.float32)[:,face,:]
		self.bbox_mat = np.array(self.bbox_mat).reshape((-1, 2, 2)).astype(np.float32)

	def __len__(self):
        	return self.length
	

	def __getitem__(self, idx):
		image = io.imread(op.join(self.img_root, self.im_list[idx]))
		bbox = self.bbox_mat[idx].copy()
		landmarks = self.lm_mat[idx].copy()
		sample = {'image': image,
			'bbox': bbox,
			'landmarks': landmarks,
			'imname': self.im_list[idx],
			'idx': idx}
		if self.transform is not None:
			self.transform(sample)
		return sample


class Occlude(object):
	def __init__(self, scale=(0.05, 0.1), prob=0.5, nbox=(3, 5)):

		if isinstance(scale, tuple):
			assert len(scale)==2
			for s in scale:
				assert s<0.5 and isinstance(s, float)
		else:
			assert isinstance(scale, float) and scale<0.5
		self.scale = scale
		
		assert isinstance(prob, float) and 0.0 <= prob <= 1.0
		self.prob = prob

		if isinstance(nbox, tuple):
			assert len(nbox)==2
			for n in nbox:
				assert isinstance(n, int)
		else:
			assert isinstance(nbox, int)
		self.nbox = nbox


	def __call__(self, sample):
		image, bbox = sample['image'], sample['bbox']
		bbox = bbox.astype(np.int32)
		if random.uniform(0, 1) < self.prob :

			w, h = bbox[1] - bbox[0] + 1
			
			if isinstance(self.nbox, int):
				nbox = self.nbox
			else:
				nbox = random.randint(self.nbox[0], self.nbox[1])

			for n in range(nbox):
				if isinstance(self.scale, float):
					scale = self.scale
				else:
					scale = random.uniform(self.scale[0], self.scale[1])

				ow, oh = int(scale*w), int(scale*h) # occlusion's size wrt bbox size
				ox, oy = random.randint(bbox[0,0], bbox[1,0]-ow), \
					 random.randint(bbox[0,1], bbox[1,1]-oh) # Top-Left of occlusion
				
				if random.randint(0,1):
					image[oy:oy+oh,ox:ox+ow,:] = +1.0 # Blacken OR Whiten the image
				else:
					image[oy:oy+oh,ox:ox+ow,:] = 0.0
	
			sample['image'] = image
		# Occlude in-place the image in memory
		return sample


class Rotate(object):
	def __init__(self, angle=15.0, prob=0.5):
		if isinstance(angle, tuple):
			assert len(angle) == 2
			for a in angle:
				assert isinstance(a, float)
		else:
			assert isinstance(angle, float)
		self.angle = angle
		
		assert isinstance(prob, float) and 0.0 <= prob <= 1.0
		self.prob = prob
	

	def __call__(self, sample):
		image, bbox, lm = sample['image'], sample['bbox'], sample['landmarks']

		if random.uniform(0,1) < self.prob:
			(h, w) = image.shape[:2]
			(cx, cy) = (w//2, h//2)

			if isinstance(self.angle, tuple):
				angle = random.uniform(self.angle[0], self.angle[1])
			else:
				angle = random.uniform(-self.angle, self.angle)
			
			M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
			cos, sin = np.abs(M[0,:2])
			nw = int( h*sin + w*cos )
			nh = int( w*sin + h*cos )
			M[0, 2] += nw/2 - cx
			M[1, 2] += nh/2 - cy
			
			image = cv2.warpAffine(image, M, (nw, nh))
			f_bbox = full_box(bbox)
			f_bbox = np.hstack((f_bbox, np.ones((f_bbox.shape[0], 1))))
			f_bbox = M.dot(f_bbox.T).T
			tl = np.min(f_bbox, axis=0)
			br = np.max(f_bbox, axis=0)
			bbox = np.vstack((tl, br)).astype(np.float32)
			lm = np.hstack((lm, np.ones((lm.shape[0], 1))))
			lm = M.dot(lm.T).T.astype(np.float32)
			sample['image'], sample['bbox'], sample['landmarks'] = image, bbox, lm
		return sample


class HFlip(object):
	def __init__(self, prob=0.5):
		assert isinstance(prob, float) and 0.0 <= prob <= 1.0
		self.prob = prob

	def __call__(self, sample):
		image, bbox, lm = sample['image'], sample['bbox'], sample['landmarks']
		if random.uniform(0, 1) < self.prob:
			image = image[:, ::-1, :]
			x = image.shape[1] - bbox[:, 0]
			bbox[:, 0] = np.array([x.min(), x.max()])
			lm[:, 0] = image.shape[1] - lm[:, 0]
			sample['image'], sample['bbox'], sample['landmarks'] = image, bbox, lm
		return sample


class Scale(object):
	def __init__(self, scale=(0.85, 1.15), prob=0.5):
                if isinstance(scale, tuple):
                        assert len(scale)==2
                        for s in scale:
                                assert isinstance(s, float)
                else:
                        assert isinstance(scale, float)
                self.scale = scale

                assert isinstance(prob, float) and 0.0 <= prob <= 1.0
                self.prob = prob
	
	def __call__(self, sample):
		bbox = sample['bbox']
		ir, ic, _ = sample['image'].shape
		if random.uniform(0,1) < self.prob:
			if isinstance(self.scale, tuple):
				scale = random.uniform(self.scale[0], self.scale[1])
			else:
				scale = random.uniform(1.0-self.scale, 1.0+self.scale)
			w, h = bbox[1] - bbox[0] + 1
			dw = w - w*scale
			dh = h - h*scale
			bbox[0] += np.array([dw/2, dh/2])
			bbox[1] -= np.array([dw/2, dh/2])
			bbox[:,0] = np.clip(bbox[:,0], 0, ic)
			bbox[:,1] = np.clip(bbox[:,1], 0, ir)
			sample['bbox'] = bbox
		return sample


class Noise(object):
	def __init__(self, m=0.0, v=0.01, prob=0.5):
		assert isinstance(m, float)
		assert isinstance(v, float)
		assert isinstance(prob, float) and 0.0 <= prob <= 1.0

		self.m = m
		self.v = v
		self.prob = prob

	def __call__(self, sample):
		image = sample['image']
		if random.uniform(0,1) < self.prob:
			image = sku.random_noise(image, mean=self.m, var=self.v)
			sample['image'] = image
		return sample


class ColorJitter(object):
        def __init__(self, bri=(0.8,1.2), con=(0.8,1.2), sat=(0.8,1.2), hue=(-0.1,0.1), prob=0.5):
            assert isinstance(bri, tuple) and len(bri)==2
            assert isinstance(con, tuple) and len(con)==2
            assert isinstance(sat, tuple) and len(sat)==2
            assert isinstance(hue, tuple) and len(hue)==2
            assert isinstance(prob, float) and 0.0 <= prob <= 1.0
            self.bri = bri
            self.con = con
            self.sat = sat
            self.hue = hue
            self.prob = prob
        
        def __call__(self, sample):
            txer = transforms.ToPILImage()
            image = txer(sample['image'])
            if random.uniform(0,1) < self.prob:
                b = random.uniform(self.bri[0], self.bri[1])
                image = txf.adjust_brightness(image, b)
                
            if random.uniform(0,1) < self.prob:
                c = random.uniform(self.con[0], self.con[1])
                image = txf.adjust_contrast(image, c)
                
            if random.uniform(0,1) < self.prob:
                s = random.uniform(self.sat[0], self.sat[1])
                image = txf.adjust_saturation(image, s)
                
            if random.uniform(0,1) < self.prob:
                h = random.uniform(self.hue[0], self.hue[1])
                image = txf.adjust_hue(image, h)
            
            sample['image'] = np.array(image)
            return sample


class toTensor(object):
        def __call__(self, sample, size=256, heat=64):
                img, bbox, lm = sample['image'], sample['bbox'], sample['landmarks']
                bbox = bbox.astype(np.int32)
                w, h = bbox[1] - bbox[0] + 1

                # Take the face using bounding box
                img = img[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]]

                # Origin at bbox top-left corner
                lm -= bbox[0]
                
                # Rescale image
                img = cv2.resize(img, (size, size))

                # Scale landmarks from bouding box to heatmap dimension
                new_lm = lm * np.array([heat/w, heat/h])

                # Initialize and create heatmaps
                hmap = np.zeros((heat,heat,5))
                for l in range(len(lm)):
                    hmap[..., l] = hm.draw_labelmap(hmap[..., l], new_lm[l], sample['imname'], sample['idx'])

                img = img.transpose((2, 0, 1)) # HWC to CHW for pyTorch
                hmap = hmap.transpose((2, 0, 1))
                
                sample['landmarks'] = lm # Just the bbox-origin landmarks

                # PyTorch does not like img[:, ::-1, ...]. Minus a tensor of zeros to workaround.
                sample['image'], sample['heatmaps'] = \
                    torch.from_numpy(img-np.zeros_like(img)).float(), torch.from_numpy(hmap).float()
                return sample


def full_box(bbox):
	x, y = zip(*bbox)
	bl = np.array([min(x), max(y)])
	tr = np.array([max(x), min(y)])
	return np.vstack((bbox, bl, tr))


def create_dataloader(root='.', batch_size=10, is_train=True, num_workers=8):
        if is_train:
                txs = transforms.Compose([ \
#		    ColorJitter(), \
#                   HFlip(), \
#		    Rotate(), \
#		    Scale(), \
#		    Noise(), \
#		    Occlude(), \
		    toTensor() \
	            ])
        else:
                txs = transforms.Compose([toTensor()])
        
        tx_dataset = FaceDataset(root=root, transform=txs, is_train=is_train)
        dataloader = DataLoader(tx_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader


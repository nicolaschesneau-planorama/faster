import sys, os, pdb
from scipy.io import loadmat
import numpy as np
from PIL import Image
import cPickle as pickle


def _prepare_dataset(path):
	with open(path + 'photo_infos.pkl','rb') as ff :
		data=pickle.load(ff)
	with open(path + 'train.txt') as ff:
		train_list=ff.readlines()
	ilist=[]
	annot=[]
	indtrain=[]
	for im in data.keys():
		boxes=[]
		yb=[]
		im_info=data[im]
		imname= str(im+'Detorted.png')
		try : facings=im_info['corr_facings']
		except : facings=im_info['facings']
		try:
			for facing in facings:
				x1=facing['x']
				y1=facing['y']
				x2=x1+facing['w']
				y2=y1+facing['h']
				boxes.append([x1,y1,x2,y2])
				yb.append(y2)
			yb=np.array(yb)
			upper_shelf=np.where(yb==np.min(yb))[0]
			for nk,box in enumerate(boxes):
				if nk in upper_shelf: box[1]+=int(0.4*(box[3]-box[1]))
				else:  box[1]+=int(0.2*(box[3]-box[1]))
		except : pass
		if len(boxes)>0:
			print imname, len(boxes)
			ilist.append(imname)
			annot.append(np.array(boxes))
			if imname +'\n' in train_list: indtrain.append(len(ilist) -1)
			#indtrain.append(len(ilist)-1)
	return ilist, annot, indtrain


class Planorama(object):
    def __init__(self):
	
        self.PATH = "/mnt/dataset/detect_GSK_2/55681a9de1d50db239964b88_55681ae9e1d50db239964c72/"
        self._pklfile = os.path.join(self.PATH, "dataset.pkl")
        if not os.path.isfile(self._pklfile):
            self.ILIST, self._annot, self._indtrain = _prepare_dataset(self.PATH)
            with open(self._pklfile,'wb') as fid:
                pickle.dump( (self.ILIST,self._annot, self._indtrain), fid)
        else:
            with open(self._pklfile,'rb') as fid:
                self.ILIST, self._annot, self._indtrain = pickle.load(fid)
        self.NIMAGES = len(self.ILIST)
        self.imname_to_index = {self.ILIST[i]: i for i in range(self.NIMAGES)}

    def get_train_indices(self):
        return self._indtrain

    def get_test_indices(self):
        return [i for i in range(len(self.ILIST)) if not i in self._indtrain]

    def get_image_file(self, x, pixelation=None):
        return os.path.join(self.PATH, self.ILIST[x] if x.__class__ == int else x)

    def get_image(self, x):
        return np.array(Image.open(self.get_image_file(x)))

    def get_resolution(self,x):
        return self.get_image(x).shape[:2]


    def get_bbox_margin(self, x, margin):
        h, w = self.get_resolution(x)      
        bbox = self._annot[ self.imname_to_index[x] if x.__class__!=int else x].astype(np.float32)
        bbox[:,0] = np.maximum(0, bbox[:,0]-margin)
        bbox[:,1] = np.maximum(0, bbox[:,1]-margin)
        bbox[:,2] = np.minimum(w-1, bbox[:,2]+margin)
        bbox[:,3] = np.minimum(h-1, bbox[:,3]+margin)
        assert np.all(bbox[:,2]>=bbox[:,0]), pdb.set_trace()
        assert np.all(bbox[:,3]>=bbox[:,1]), pdb.set_trace()
        return bbox

    def get_bbox(self,x):
        return self.get_bbox_margin(x, margin=0)


    def show_boxes(self, x):
        import matplotlib.pylab as plt
        plt.ion()
        plt.clf()
        plt.imshow(self.get_image(x))
        boxes = self.get_bbox(x)
        for i in range(boxes.shape[0]):
            plt.plot( boxes[i,np.array([0,0,2,2,0],dtype=np.int32)],  boxes[i,np.array([1,3,3,1,1],dtype=np.int32)], 'y', lw=3)
        
    def show_boxes_html(self, image_path="images/"):
        # too many images on the same page => need to draw one page each 1000 images + one main page
        def gen_bbox_html(imname, boxes, hmtlid, resolution, colors=["rgba(255,0,0,0.5)"],flip=False):
            if boxes.ndim == 1: boxes = boxes[None,:]
            h,w = resolution
            strflip = "filter:FlipH; -ms-filter:FlipH; -moz-transform: scaleX(-1);-webkit-transform: scaleX(-1); -o-transform: scaleX(-1); transform: scaleX(-1);" if flip else ""
            s = '<div id="div_{:s}" style="width: {:d}px; height: {:d}px; background-image:{:s}; background-size: {:d}px auto; background-repeat:no-repeat; {:s}">'.format(hmtlid, w, h, "url("+imname.split('/')[-1]+")", w,strflip)
            s += '<canvas id="{:s}" height="{:d}" width="{:d}" style="cursor: crosshair">'.format(hmtlid, h, w)
            s += "<script>"
            s += "var canvas = document.getElementById('{:s}');".format(hmtlid)
            s += "var context = canvas.getContext('2d');"
            for i in range(boxes.shape[0]):
                if i<len(colors):
                    s += "context.fillStyle = '%s';"%(colors[i])
                bb = [int(boxes[i,0]), int(boxes[i,1]), int(boxes[i,2]-boxes[i,0]), int(boxes[i,3]-boxes[i,1])]
                #s += "context.fillRect(%d,%d,%d,%d);"%(bb[0],bb[1],bb[2],bb[3] )
		s += "context.lineWidth='15';"
		s += "context.strokeStyle='green';"
		s += "context.rect(%d,%d,%d,%d);"%(bb[0],bb[1],bb[2],bb[3] )
		s += "context.stroke();"
            s += "</script>"
            s += "</canvas>"
            s += "</div>"
            return s
        from html import HTML, Table, htmloptions
        doc = HTML()
        h = doc.header()
        h.title('MPII Human Pose - boxes')
        #h.link(rel="shortcut icon", href="http://lear.inrialpes.fr/image/lear2.ico")
        #h.link(rel="stylesheet", type="text/css", href="http://pascal.inrialpes.fr/data2/pweinzae/xp/lear.css")    
        h.script(src="https://ajax.googleapis.com/ajax/libs/jquery/1.4.4/jquery.min.js", type="text/javascript")
        script="""
        function select_range(){
            range = document.getElementById('select_range').value;
            $("#div_display").load("boxes_"+range+".html");
        }
        """
        h.script(text=script, **{"type":"text/javascript", "language":"javascript"})   
        b = doc.body(onload="select_range()")
        b.h(1, ' a Plano dataset test (%d images with a box)'%(self.NIMAGES))
        b.p('Boxes are computed to encompass all products, with a margin of 0 pixels.')
        b.p('<select id="select_range" onChange="select_range()">%s</select>'%(htmloptions(["%05d_%05d"%(i*100+1, min(i*100+100, self.NIMAGES)) for i in range(0,self.NIMAGES//100)])))
        b.div(id="div_display",style="zoom: 1;")
        doc.save( os.path.join(self.PATH, "boxes.html") )
        for j in range(0,self.NIMAGES//100):
            t = Table(border="1", cellpadding="5")
            imin = j*100;
            imax = min(j*100+100, self.NIMAGES)
            for i in range(imin,imax):
                if i%5==0: r = t.row(align="center")
                r.cell( gen_bbox_html( image_path+self.ILIST[i], self.get_bbox(i), self.ILIST[i][:-4], self.get_resolution(i)) )
            with open( os.path.join(self.PATH,"boxes_%05d_%05d.html"%(imin+1, imax)),'w') as fid:
                fid.write(t.tostr())

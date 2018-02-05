import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list,get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import pdb
import cPickle
from html import HTML, Table, htmloptions
import cv2
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

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
    
def display(net, imdb, max_per_image=100, thresh=0.05,vis=False):

    output_dir = get_output_dir(imdb, net)
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'rb') as f:
        all_boxes= cPickle.load(f)
    all_boxes=all_boxes[1]    
    num_images = len(imdb.image_index)
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
    b.h(1, ' a Plano dataset test (%d images with a box)'%(num_images))
    b.p('Boxes are computed on the test split')
    display_path='/'.join(imdb.image_path_at(0).split('/')[:-1])
    doc.save(os.path.join(display_path,'display_results.html'))

    t=Table(border="1",cellpadding="5")
    for i in xrange(num_images):
        if i%5==0: r = t.row(align='center')
        boxes=all_boxes[i]
        impath=imdb.image_path_at(i)
        im = cv2.imread(impath)
        resolution=im.shape[:2]
        r.cell(gen_bbox_html(impath,boxes,impath.split('/')[-1],resolution))
    
    with open(os.path.join(display_path,'display_results.html'),'w') as fid:
            fid.write(t.tostr())


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    display(net, imdb, max_per_image=args.max_per_image, vis=args.vis)


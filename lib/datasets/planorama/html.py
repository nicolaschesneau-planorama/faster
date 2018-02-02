import os
from PIL import Image

# see help for ocmmon HTML tags at http://www.mountaindragon.com/html/text.htm


class Node:
  def __init__(self,tag,text='',props=dict()):
    self.children = []
    self.tag=tag
    self.text=text
    self.props=props
  def add(self, node):
    self.children.append(node)
    return node
  def tostr(self):
    s = ""
    if not self.props:
      s+= "<%s>%s"%(self.tag,self.text)
    else:
      s+= "<%s %s>%s"%(self.tag,' '.join(["%s='%s'"%(k,v) for k,v in self.props.items() if v!=None]),self.text)
    for child in self.children:
      s += child.tostr()
    s += "</%s>"%self.tag
    return s
  def write(self,fout):
    if not self.props:
      print >>fout, "<%s>%s"%(self.tag,self.text)
    else:
      print >>fout, "<%s %s>%s"%(self.tag,' '.join(["%s='%s'"%(k,v) for k,v in self.props.items() if v!=None]),self.text)
    for child in self.children:
      child.write(fout)
    print >>fout, "</%s>"%self.tag
  def first(self,tag,order=1):
    if self.tag==tag: return self
    for c in self.children[::order]:
      res = c.first(tag,order)
      if res: return res
    return None
  def last(self,tag):
    return self.first(tag,-1)



class HTML (Node):
  def __init__(self):
    Node.__init__(self,'html')
  def header(self,**kw):
    return self.add(Header(**kw))
  def body(self,**kw):
    return self.add(BodyNode('body',props=kw))
  def save(self,fname):
    fout = open(fname,'w') if type(fname)==str else fname
    for e in self.children:
      e.write(fout)
  def show(self,fname=''):
    if not fname:  fname = os.tmpname()+'.html' 
    self.save(fname)
    os.system('/opt/google/chrome/google-chrome '+fname)



class Header (HTML):
  def __init__(self, **kw):
    Node.__init__(self,'header',props=kw)
  def title(self,text):
    return self.add(Node('title',text=text))
  def script(self, text="", **kw):
    return self.add(Node('script',text=text, props=kw))
  def link(self, **kw):
    return self.add(Node('link', props=kw))
  def meta(self):
    return self.add(Node('meta', props={"http-equiv":"Content-Type", "content": "charset=iso-8859-1"}))

class BodyNode (Node):
  # title of section
  def h(self, strength, text='', **kw):
    return self.add(BodyNode('h%d'%strength, text=text, props=kw))
  # paragraph
  def p(self, text='', **kw):
    return self.add(BodyNode('p',text=text, props=kw))
  # bold
  def bold(self, text='', **kw):
    return self.add(BodyNode('b',text=text, props=kw))
  def b(self, text='', **kw):
    return self.add(BodyNode('b',text=text, props=kw))
  # italic
  def italic(self, text='', **kw):
    return self.add(BodyNode('i',text=text, props=kw))
  def i(self, text='', **kw):
    return self.add(BodyNode('i',text=text, props=kw))
  # span/text
  def span(self, text='', **kw):
    return self.add(BodyNode('span',text=text, props=kw))
  # font
  def font(self,text='',color=None,face=None,size=None):
    return self.add(BodyNode('font',text=text, props={'color':color,'face':face,'size':size}))
  # small
  def small(self, text='', **kw):
    return self.add(BodyNode('small',text=text, props=kw))
  def big(self, text='', **kw):
    return self.add(BodyNode('big',text=text, props=kw))
  # centered 
  def center(self, text='', **kw):
    return self.add(BodyNode('center',text=text, props=kw))
  # div
  def div(self, text='', **kw):
    return self.add(BodyNode('div', text=text, props=kw))
  # unordered list
  def unordlist(self, text='', **kw):
    return self.add(BodyNode('ul', text=text, props=kw))
  # ordered list
  def ordlist(self, text='', **kw):
    return self.add(BodyNode('ol', text=text, props=kw))
  def item(self, text='', type=None, **kw):
    kw['type'] = type # non-ord {'circle', 'square', 'disc'}, ord  {'1', 'A', 'a', 'I', 'i'}
    return self.add(BodyNode('li', text=text, props=kw))
  # line break
  def br(self):
    self.add(Node('br'))
  # horizontal line
  def hr(self):
    self.add(Node('hr'))
  # table
  def table(self, **kw):
    return self.add(Table(**kw))
  # image
  def image(self, img, **kw):
    return self.add(Image(img,**kw))
  # link
  def a(self, href, text='', **kw):
    kw['href'] = href
    return self.add(BodyNode('a', text=text, props=kw))
  def hidden(self, text, **kw):
    kw['type'] = 'hidden'
    kw['value'] = text
    return self.add(BodyNode('input',props=kw))
  def imagelink(self, img, **kw):
    return self.add( BodyNode('a', text=Image(img,**kw).tostr(), props={"href":img}) )
  
class Table (Node):
  def __init__(self,**kw):
    Node.__init__(self,'table',props=kw)
  def row(self,elems=[],header=False,**kw):
    r=TableRow(header, **kw)
    for e in elems:
      if issubclass(e.__class__,Node):
        r.add(e)
      else: 
        r.cell(str(e))
    return self.add(r)
  def fromlist(self, elems, header=None):
    if header and type(header)!=bool: elems=[header]+elems; header=True
    for row in elems:
      self.row(row,header=header)
      header=False  # only once


class TableRow (Node):
  def __init__(self, isheader=False, **kw):
    Node.__init__(self,'tr',props=kw)
    self.isheader=isheader
  def cell(self, text='', **kw):
    return self.add(BodyNode(self.isheader and 'th' or 'td',text=text,props=kw))



class Image (Node):
  def __init__(self, img, dir='', name='', width=None, height=None, alt=None):
    if type(img)==str:  loc = img
    else:
      if name:
        img.save(os.path.join(dir,name))
        loc = name
      else:
        loc =  os.tmpnam()+".png"
        img.save(loc)
    Node.__init__(self,'img',props={'src':loc,'width':width,'height':height,'alt':alt,'title':alt})


def htmlspace(n):
    return "&nbsp;".join(["" for i in range(n)])
def htmloptions(l):
    return "".join(["<option>"+s+"</option>" for s in l])


if __name__=='__main__':
  import pdb
  
  doc = HTML()
  doc.header().title('test of python-generated HTML page')
  body=doc.body()
  body.h(1,"1. Title of page")
  body.p('a paragraph of text')
  body.h(2,"2.1 second title")
  p=body.p()
  p.italic('another')
  p.font(color='red').bold('paragraph')
  p.span('of text')
  body.h(3,'2.1.1. sub-sub-title')
  body.p("Here is a list:")
  ls=body.unordlist()
  ls.item("first item")
  ls.item("second item")
  ls.item("final item")
  body.hr()
  body.table(border=1).fromlist([[1,2],[3,4]],header=['col1','col2'])
  body.br()
  body.center().image(img='/home/lear/revaud/coca-cola.jpg',width=500,height=300)
  body.hr()
  tab=body.table(border=0)
  tab.row(['coca-cola']*5,header=True)
  for i in range(3):
    r = body.last('table').row()
    for j in range(5):
      r.cell(bgcolor=['#00FF00','red'][(i+j)%2]).image('/home/lear/revaud/coca-cola2.jpg',width=200)
  
  doc.show('/tmp/test.html')
  print 'result stored in /tmp/test.html'


































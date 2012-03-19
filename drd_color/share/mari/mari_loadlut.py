import mari

LUT3D_SIZE=32 

filter_collection=None
saved_filter_collection=None

def storePostFilterCollection():
    global saved_filter_collection
    saved_filter_collection = mari.gl_render.currentPostFilterCollection()

def restorePostFilterCollection():
    global saved_filter_collection
    mari.gl_render.setPostFilterCollection(saved_filter_collection)

def loadLUT(lutfile, filtername):

    global filter_collection

    desc = "sampler3D lut3d_ocio_$ID_;\n\nvec4 display_ocio_$ID_(in vec4 inPixel, const uniform sampler3D lut3d)\n{    vec4 out_pixel = inPixel;\n    out_pixel.rgb = max(vec3(1.17549e-38, 1.17549e-38, 1.17549e-38), vec3(1, 1, 1) * out_pixel.rgb + vec3(0, 0, 0));\n    out_pixel.rgb = vec3(1.4427, 1.4427, 1.4427) * log(out_pixel.rgb) + vec3(0, 0, 0);\n    out_pixel = vec4(0.047619, 0.047619, 0.047619, 1) * out_pixel;\n    out_pixel = vec4(0.714286, 0.714286, 0.714286, 0) + out_pixel;\n    out_pixel.rgb = texture3D(lut3d, 0.96875 * out_pixel.rgb + 0.015625).rgb;\n    return out_pixel;\n}"

    body = "{ Out = display_ocio_$ID_(Out, lut3d_ocio_$ID_); }"

    name = filtername

    postfilter = filter_collection.createGLSL(name, desc, body)
    storePostFilterCollection()
    mari.gl_render.setPostFilterCollection(filter_collection)

    fp = open(lutfile,'r')
    if fp==None:
        print lutfile + " not found"

    #print fp
    lut3d = []
    
    line=fp.readline()
    values=line.split()

    for item in values:
        lut3d.append(float(item))
                
    fp.close()

    postfilter.setTexture3D("lut3d_ocio_$ID_", LUT3D_SIZE, LUT3D_SIZE, LUT3D_SIZE, postfilter.FORMAT_RGB, lut3d)
    mari.gl_render.setPostFilterCollection(filter_collection)


filter_collection = mari.gl_render.findPostFilterCollection( 'HF2 Color Look-up' )
if None is filter_collection:
    filter_collection = mari.gl_render.createPostFilterCollection( 'HF2 Color Look-up' )

filter_collection.clear()
pb=mari.canvases.paintBuffer()
pb.setClampColors(0)
#print "Clamp colors: %d" % pb.clampColors()

calibconf=os.environ.get("HOME")+'/.config/calib.yaml'

#print calibconf
dreamcolor=0

if os.path.exists(calibconf):
    conf_fp=open(calibconf,'r')
    line=conf_fp.readline()
    line=conf_fp.readline()
    #print line
    if line.find("DCI")!=-1:
        dreamcolor=1
    conf_fp.close()

if dreamcolor==1:
    print "Loading Dreamcolor LUT..."
    lutfile=os.environ.get("DRD_MARILUT_DREAMCOLOR")
else:
    print "Loading sRGB LUT..."
    lutfile=os.environ.get("DRD_MARILUT_SRGB")

print lutfile
    
loadLUT(lutfile, "HF2 Dreamcolor" if dreamcolor == 1 else "HF2 sRGB")

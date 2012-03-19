import sys
import yaml
import PyOpenColorIO as OCIO
sys.stdout.write("DRD INFO: loaded drd_color\n")

# Load generic 'Project' viewer
#nuke.ViewerProcess.register( 'Project', nuke.Node, ( 'ProjectViewer', ''))
   # Formats the display and transform, e.g "Film1D (sRGB"
DISPLAY_UI_FORMAT = "%(transform)s (%(display)s)"
#
cfg = OCIO.GetCurrentConfig()


#nuke.ViewerProcess.register(
#                            name = DISPLAY_UI_FORMAT % {"transform": "Project", "display": "Dreamcolor"},
#                            call = nuke.nodes.ProjectViewer,
#                            args = (),
#                            kwargs = {"device": "Dreamcolor", "transform": "Film"})

#nuke.ViewerProcess.register(
#                            name = DISPLAY_UI_FORMAT % {"transform": "Project", "display": "sRGB"},
#                            call = nuke.nodes.OCIODisplay,
#                            args = (),
#                            kwargs = {"device": "sRGB", "transform": "Film"})

_file_path = os.environ['HOME']+"/.config/calib.yaml"
defaultDisplay='Project (sRGB)'
defaultXform='Project'
if os.path.exists(_file_path) :
    stream = open(_file_path, 'r')
    ybdlFile = yaml.load(stream)
    if ybdlFile['target'] == 'DCI' :
        defaultDisplay='Project (Dreamcolor)'
#
#nuke.knobDefault("Viewer.viewerProcess", DISPLAY_UI_FORMAT % {'transform': defaultXform, "display": defaultDisplay})

#print "Default display: %s" % defaultDisplay

nuke.ViewerProcess.register( 'Project (Dreamcolor)', nuke.Node, ( 'ProjectViewer_Dreamcolor', ''))
nuke.ViewerProcess.register( 'Project (sRGB)', nuke.Node, ( 'ProjectViewer_sRGB', ''))
nuke.knobDefault( 'Viewer.viewerProcess', defaultDisplay )

#nd=nuke.ViewerProcess.node("Dreamcolor").

